import axolotl.logging_config

axolotl.logging_config.configure_logging()


import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import fire
import yaml
from axolotl.cli.merge_lora import do_cli as axolotl_merge_lora_cli
from axolotl.cli.train import do_cli as axolotl_train_cli
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import barrier, is_main_process, zero_first
from axolotl.utils.models import load_tokenizer
from rich import console, panel
from transformers.utils import is_torch_bf16_gpu_available, is_torch_tf32_available

from checkpoint_utils import (
    cleanup_checkpoints,
    get_last_checkpoint_for_resume_if_any,
    get_step_for_final_model,
)
from data_utils import dataset_uri_to_axolotl_datasources
from mlfoundry_utils import (
    get_or_create_run,
    log_model_to_mlfoundry,
    maybe_log_params_to_mlfoundry,
    sanitize_name,
)
from utils import (
    maybe_set_custom_tempdir,
    maybe_set_torch_max_memory,
    temporarily_unset_distributed_envs,
    try_cleanup_gpus,
)

logger = logging.getLogger("axolotl")

# CURRENT LIMITATIONS
# Axolotl sets report_to to None instead of "none"
# There should be an option to add only missing special tokens
# Have to hack axolotl module globals to hook our own code
# micro batch size still needs to be decided by the user. 1 is okay because we are using sample packing now


TFY_INTERNAL_JOB_NAME = os.getenv("TFY_INTERNAL_COMPONENT_NAME")
TFY_INTERNAL_JOB_RUN_NAME = os.getenv("TFY_INTERNAL_JOB_RUN_NAME")
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


def set_cfg_option_if_auto(cfg, key, value, force=False):
    if cfg[key] in ("auto", None) or force:
        logger.info(f"`{key}` is being automatically set to `{value}`")
        cfg[key] = value


def load_config_file(path):
    with open(path, encoding="utf-8") as file:
        cfg = DictDefault(yaml.safe_load(file))
    return cfg


def make_axolotl_config(config_base, kwargs, timestamp=None):
    cfg = load_config_file(path=config_base)
    cfg_keys = cfg.keys()
    # TODO: Support nested overriding via kwargs: --a.b.c or --a.0.b
    for k, _ in kwargs.items():
        # if not strict, allow writing to cfg even if it's not in the yml already
        if k in cfg_keys or not cfg.strict:
            # handle booleans
            if isinstance(cfg[k], bool):
                cfg[k] = bool(kwargs[k])
            else:
                cfg[k] = kwargs[k]
    if not cfg.output_dir:
        raise ValueError("`output_dir` must be set in config base")

    if is_main_process():
        if cfg.cleanup_output_dir_on_start is True:
            logger.warning(f"--cleanup_output_dir_on_start was to set to True, wiping {cfg.output_dir}")
            if os.path.exists(cfg.output_dir):
                shutil.rmtree(cfg.output_dir)

    data_dir = os.path.join(os.path.abspath(cfg.output_dir), "data")
    set_cfg_option_if_auto(cfg, "data_dir", data_dir)
    cfg.output_dir = os.path.join(os.path.abspath(cfg.output_dir), "model")
    axolotl_config = os.path.join(cfg.output_dir, "axolotl_config.yaml")

    if is_main_process():
        import torch

        set_cfg_option_if_auto(cfg, "tokenizer_config", cfg.base_model_config or cfg.base_model)

        os.makedirs(cfg.data_dir, exist_ok=True)
        os.makedirs(cfg.output_dir, exist_ok=True)

        run = None
        if cfg.truefoundry_ml_enable_reporting is True:
            if TFY_INTERNAL_JOB_RUN_NAME:
                fallback_run_name = f"finetune-{sanitize_name(TFY_INTERNAL_JOB_RUN_NAME)}"
            else:
                fallback_run_name = f"finetune-{timestamp}"
            set_cfg_option_if_auto(cfg, "truefoundry_ml_run_name", fallback_run_name)

            run = get_or_create_run(
                ml_repo=cfg.truefoundry_ml_repo,
                run_name=cfg.truefoundry_ml_run_name,
                auto_end=False,
            )

            if cfg.truefoundry_ml_log_checkpoints is True:
                if TFY_INTERNAL_JOB_RUN_NAME:
                    truefoundry_ml_checkpoint_artifact_name = f"ckpt-{sanitize_name(TFY_INTERNAL_JOB_RUN_NAME)}"
                else:
                    truefoundry_ml_checkpoint_artifact_name = f"ckpt-{run.run_name}"
                set_cfg_option_if_auto(
                    cfg,
                    "truefoundry_ml_checkpoint_artifact_name",
                    truefoundry_ml_checkpoint_artifact_name,
                )
            else:
                cfg.truefoundry_ml_log_checkpoints = False
                cfg.truefoundry_ml_checkpoint_artifact_name = None

        if cfg.resume_from_checkpoint == "auto":
            resume_from_checkpoint = True
        else:
            resume_from_checkpoint = cfg.resume_from_checkpoint
        last_checkpoint_dir = get_last_checkpoint_for_resume_if_any(
            output_dir=cfg.output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
            mlfoundry_enable_reporting=cfg.truefoundry_ml_enable_reporting,
            mlfoundry_ml_repo=cfg.truefoundry_ml_repo,
            mlfoundry_checkpoint_artifact_name=cfg.truefoundry_ml_checkpoint_artifact_name,
        )
        cfg.resume_from_checkpoint = last_checkpoint_dir

        set_cfg_option_if_auto(cfg, "eval_steps", 0.1)
        set_cfg_option_if_auto(cfg, "save_steps", 0.1)

        is_ampere_or_newer = torch.cuda.get_device_capability(device=LOCAL_RANK) >= (
            8,
            0,
        )
        is_tf32_supported = is_ampere_or_newer and is_torch_tf32_available()
        is_bf16_supported = is_ampere_or_newer and is_torch_bf16_gpu_available()
        set_cfg_option_if_auto(cfg, "tf32", is_tf32_supported)
        # TODO: Axolotl doesn't seem to do anything differently even though it says setting bfloat16/float16 will disable AMP
        set_cfg_option_if_auto(cfg, "bf16", is_bf16_supported)
        set_cfg_option_if_auto(cfg, "bfloat16", is_bf16_supported)
        set_cfg_option_if_auto(cfg, "fp16", not is_bf16_supported)
        set_cfg_option_if_auto(cfg, "float16", not is_bf16_supported)

        set_cfg_option_if_auto(cfg, "flash_attention", is_ampere_or_newer)
        set_cfg_option_if_auto(cfg, "flash_attn_cross_entropy", is_ampere_or_newer)
        set_cfg_option_if_auto(cfg, "flash_attn_rms_norm", is_ampere_or_newer)

        set_cfg_option_if_auto(cfg, "load_in_4bit", cfg.adapter == "qlora")
        set_cfg_option_if_auto(cfg, "flash_attn_fuse_mlp", cfg.adapter not in {"qlora", "lora"})
        set_cfg_option_if_auto(cfg, "flash_attn_fuse_qkv", cfg.adapter not in {"qlora", "lora"})

        use_unsloth = False  # torch.cuda.device_count() == 1
        set_cfg_option_if_auto(cfg, "unsloth_cross_entropy_loss", use_unsloth)
        set_cfg_option_if_auto(cfg, "unsloth_lora_mlp", use_unsloth)
        set_cfg_option_if_auto(cfg, "unsloth_lora_qkv", use_unsloth)
        set_cfg_option_if_auto(cfg, "unsloth_lora_o", use_unsloth)
        set_cfg_option_if_auto(cfg, "unsloth_rms_norm", use_unsloth)
        set_cfg_option_if_auto(cfg, "unsloth_rope", use_unsloth)

        if cfg.datasets == "auto":
            if not cfg.train_data_uri:
                raise ValueError("`train_data_uri` cannot be null when set to `datasets` is set to auto")
            cfg.datasets = dataset_uri_to_axolotl_datasources(
                uri=cfg.train_data_uri,
                download_dir=cfg.data_dir,
                dataset_type=cfg.dataset_type,
            )
        if cfg.test_datasets == "auto":
            if cfg.val_data_uri and str(cfg.val_data_uri).lower() != "na":
                cfg.test_datasets = dataset_uri_to_axolotl_datasources(
                    uri=cfg.val_data_uri,
                    download_dir=cfg.data_dir,
                    dataset_type=cfg.dataset_type,
                )
            elif cfg.val_set_size:
                set_cfg_option_if_auto(cfg, "test_datasets", None, force=True)
            else:
                raise ValueError(
                    "At least one of `val_data_uri` or `val_set_size` must be non null when `test_datasets` is set to auto"
                )

        if cfg.test_datasets:
            set_cfg_option_if_auto(cfg, "val_set_size", 0, force=True)

        # TODO: Upload processed data to resume from
        set_cfg_option_if_auto(cfg, "resume_from_checkpoint", None)

        # TODO: Figure if we should mess around and add special tokens
        # Problem is axolotl tries fixing/adding some tokens by its own.
        # We don't want to override those decisions without understanding the consequences
        set_cfg_option_if_auto(cfg, "special_tokens", {})
        tokenizer = load_tokenizer(cfg=cfg)
        if not tokenizer.pad_token:
            cfg["special_tokens"]["pad_token"] = tokenizer.eos_token
        set_cfg_option_if_auto(cfg, "lora_modules_to_save", [])
        logger.info(f"Prepared config: {cfg}")
        # This hack is needed because yaml dump refuses to treat DictDefault as dict
        yaml.add_representer(
            DictDefault,
            lambda dumper, data: dumper.represent_mapping("tag:yaml.org,2002:map", data.items()),
        )
        print(f"Saving axolotl config to {axolotl_config}")
        with open(axolotl_config, "w") as f:
            yaml.dump(cfg, f)

        if run:
            maybe_log_params_to_mlfoundry(run, cfg)
    return axolotl_config


def _train_with_truefoundry(config_base: Path = Path("examples/"), **kwargs):
    maybe_set_custom_tempdir()
    maybe_set_torch_max_memory(device=LOCAL_RANK)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    with zero_first(is_main_process()):
        axolotl_config = make_axolotl_config(
            config_base=config_base,
            kwargs=kwargs,
            timestamp=timestamp,
        )
    barrier()
    axolotl_train_cli(config=axolotl_config)
    barrier()
    logger.info("Clearing gpus before moving ahead ...")
    try_cleanup_gpus()
    barrier()
    if is_main_process():
        cfg = load_config_file(path=axolotl_config)
        model_dir = cfg.output_dir
        log_step = get_step_for_final_model(
            output_dir=cfg.output_dir, load_best_model_at_end=cfg.load_best_model_at_end
        )
        cleanup_checkpoints(output_dir=cfg.output_dir)
        if cfg.adapter in {"lora", "qlora"} and cfg.merge_adapters_post_train:
            with temporarily_unset_distributed_envs():
                axolotl_merge_lora_cli(config=axolotl_config, device_map="auto")
            model_dir = os.path.join(model_dir, "merged")
            model_parent_dir = os.path.dirname(model_dir)
            # Copy tensorboard logs
            tensorboard_logs_dir = os.path.join(model_parent_dir, "runs")
            if os.path.exists(tensorboard_logs_dir):
                shutil.copytree(
                    tensorboard_logs_dir,
                    os.path.join(model_dir, "runs"),
                    dirs_exist_ok=True,
                )
            # Copy axolotl config
            if os.path.exists(axolotl_config):
                shutil.copy2(axolotl_config, os.path.join(model_dir, "axolotl_config.yaml"))
            # Copy README.md
            readme_path = os.path.join(model_parent_dir, "README.md")
            if os.path.exists(readme_path):
                shutil.copy2(readme_path, os.path.join(model_dir, "README.md"))
            logger.info(f"Merged model has been saved to {model_dir}")
        if cfg.truefoundry_ml_enable_reporting is True and cfg.truefoundry_ml_log_merged_model is True:
            *_, model_name = cfg.base_model.rsplit("/", 1)
            model_name = "-".join(["finetuned", model_name, timestamp])
            model_name = sanitize_name(model_name)
            run = get_or_create_run(
                ml_repo=cfg.truefoundry_ml_repo,
                run_name=cfg.truefoundry_ml_run_name,
                auto_end=False,
            )
            log_model_to_mlfoundry(
                run=run,
                model_name=model_name,
                model_dir=model_dir,
                hf_hub_model_id=cfg.base_model,
                metadata={},
                step=log_step,
            )
            run.end()


def train_with_truefoundry(config_base: Path = Path("examples/"), **kwargs):
    try:
        _train_with_truefoundry(config_base=config_base, **kwargs)
    except Exception as e:
        c = console.Console()
        error_message = (
            f"Rank {LOCAL_RANK} failed with error: {str(e)}\nPlease see the following traceback for more details."
        )
        logger.error(error_message)
        c.print(panel.Panel.fit(f"[red]{error_message}[/]", title="Error", border_style="bright_red"))
        raise


if __name__ == "__main__":
    fire.Fire(train_with_truefoundry)
