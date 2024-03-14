from monkey_patch import monkey_patch_axolotl_internals

monkey_patch_axolotl_internals()

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
from transformers.utils import is_torch_bf16_gpu_available, is_torch_tf32_available

from checkpoint_utils import cleanup_checkpoints, get_last_checkpoint_for_resume_if_any
from data_utils import dataset_uri_to_axolotl_datasources
from mlfoundry_utils import (
    get_or_create_run,
    log_model_to_mlfoundry,
    maybe_log_params_to_mlfoundry,
    sanitize_name,
)
from utils import maybe_set_custom_tempdir, maybe_set_torch_max_memory, try_cleanup_gpus

logger = logging.getLogger("axolotl")

# CURRENT LIMITATIONS
# Axolotl sets report_to to None instead of "none"
# There should be an option to add only missing special tokens
# Cannot control truncation vs dropping when data exceeds sequence length
# Have to hack axolotl module globals to hook our own code
# micro batch size still needs to be decided by the user. 1 is okay because we are using sample packing now


TFY_INTERNAL_JOB_NAME = os.getenv("TFY_INTERNAL_COMPONENT_NAME")
TFY_INTERNAL_JOB_RUN_NAME = os.getenv("TFY_INTERNAL_JOB_RUN_NAME")
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


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
        os.makedirs(cfg.data_dir, exist_ok=True)
        os.makedirs(cfg.output_dir, exist_ok=True)

        run = None
        if cfg.mlfoundry_enable_reporting is True:
            if TFY_INTERNAL_JOB_RUN_NAME:
                fallback_run_name = f"finetune-{sanitize_name(TFY_INTERNAL_JOB_RUN_NAME)}"
            else:
                fallback_run_name = f"finetune-{timestamp}"
            set_cfg_option_if_auto(cfg, "mlfoundry_run_name", fallback_run_name)

            run = get_or_create_run(
                ml_repo=cfg.mlfoundry_ml_repo,
                run_name=cfg.mlfoundry_run_name,
                auto_end=False,
                create_ml_repo=False,
            )

            if cfg.mlfoundry_log_checkpoints is True:
                if TFY_INTERNAL_JOB_RUN_NAME:
                    mlfoundry_checkpoint_artifact_name = f"ckpt-{sanitize_name(TFY_INTERNAL_JOB_RUN_NAME)}"
                else:
                    mlfoundry_checkpoint_artifact_name = f"ckpt-{run.run_name}"
                set_cfg_option_if_auto(cfg, "mlfoundry_checkpoint_artifact_name", mlfoundry_checkpoint_artifact_name)
            else:
                cfg.mlfoundry_log_checkpoints = False
                cfg.mlfoundry_checkpoint_artifact_name = None

        if cfg.resume_from_checkpoint == "auto":
            resume_from_checkpoint = True
        else:
            resume_from_checkpoint = cfg.resume_from_checkpoint
        last_checkpoint_dir = get_last_checkpoint_for_resume_if_any(
            output_dir=cfg.output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
            mlfoundry_enable_reporting=cfg.mlfoundry_enable_reporting,
            mlfoundry_ml_repo=cfg.mlfoundry_ml_repo,
            mlfoundry_checkpoint_artifact_name=cfg.mlfoundry_checkpoint_artifact_name,
        )
        cfg.resume_from_checkpoint = last_checkpoint_dir

        set_cfg_option_if_auto(cfg, "eval_steps", 0.1)
        set_cfg_option_if_auto(cfg, "save_steps", 0.1)
        set_cfg_option_if_auto(cfg, "tf32", is_torch_tf32_available())
        # TODO: Axolotl doesn't seem to do anything differently even though it says setting bfloat16/float16 will disable AMP
        set_cfg_option_if_auto(cfg, "bf16", is_torch_bf16_gpu_available())
        set_cfg_option_if_auto(cfg, "bfloat16", is_torch_bf16_gpu_available())
        set_cfg_option_if_auto(cfg, "fp16", not is_torch_bf16_gpu_available())
        set_cfg_option_if_auto(cfg, "float16", not is_torch_bf16_gpu_available())

        set_cfg_option_if_auto(cfg, "load_in_4bit", cfg.adapter == "qlora")
        set_cfg_option_if_auto(cfg, "flash_attn_fuse_mlp", cfg.adapter not in {"qlora", "lora"})
        set_cfg_option_if_auto(cfg, "flash_attn_fuse_qkv", cfg.adapter not in {"qlora", "lora"})

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
        set_cfg_option_if_auto(cfg, "lora_modules_to_save", [])

        logger.info(f"Prepared config: {cfg}")
        # This hack is needed because yaml dump refuses to tread DictDefault as dict
        yaml.add_representer(
            DictDefault, lambda dumper, data: dumper.represent_mapping("tag:yaml.org,2002:map", data.items())
        )
        print(f"Saving axolotl config to {axolotl_config}")
        with open(axolotl_config, "w") as f:
            yaml.dump(cfg, f)

        if run:
            maybe_log_params_to_mlfoundry(run, cfg)
    return axolotl_config


def train_with_truefoundry(config_base: Path = Path("examples/"), **kwargs):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    maybe_set_custom_tempdir()
    maybe_set_torch_max_memory(device=local_rank)
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
        cleanup_checkpoints(output_dir=cfg.output_dir)
        if cfg.adapter in {"lora", "qlora"}:
            axolotl_merge_lora_cli(config=axolotl_config)
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
        if cfg.mlfoundry_enable_reporting is True:
            *_, model_name = cfg.base_model.rsplit("/", 1)
            model_name = "-".join(["finetuned", model_name, timestamp])
            model_name = sanitize_name(model_name)
            run = get_or_create_run(
                ml_repo=cfg.mlfoundry_ml_repo,
                run_name=cfg.mlfoundry_run_name,
                auto_end=False,
                create_ml_repo=False,
            )
            log_model_to_mlfoundry(
                run=run,
                model_name=model_name,
                model_dir=model_dir,
                hf_hub_model_id=cfg.base_model,
                metadata={},
                # TODO (chiragjn): Need to add step here to link with metrics!
            )
            run.end()


if __name__ == "__main__":
    fire.Fire(train_with_truefoundry)


# def check_if_model_will_fit_only_with_gpus(
#     model_id: str,
#     revision: Optional[str],
#     torch_dtype,
# ):
#     with init_empty_weights():
#         config = AutoConfig.from_pretrained(
#             model_id,
#             revision=revision,
#             trust_remote_code=True,
#         )
#         model = AutoModelForCausalLM.from_config(
#             config=config,
#             trust_remote_code=True,
#             torch_dtype=torch_dtype,
#             # low_cpu_mem_usage=True,
#         )
#     device_map = infer_auto_device_map(model, dtype=torch_dtype)
#     logger.info(f"Inferred device_map for auto settings: {device_map}")
#     if any(not isinstance(v, int) for v in device_map.values()):
#         raise RuntimeError(
#             "For lora/qlora the model must entirely fit on gpus without any kind of offloading to prevent bugs with merging! "
#             "With the current configuration model is being offloaded to cpu/disk. This causes incorrect model saving. See https://github.com/huggingface/peft/issues/868"
#         )
