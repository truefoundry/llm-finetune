import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import fire
import yaml
from axolotl.cli.merge_lora import do_cli as axolotl_merge_lora_cli
from axolotl.cli.train import do_cli as axolotl_train_cli
from axolotl.core.trainer_builder import AxolotlTrainer
from axolotl.utils.callbacks import GPUStatsCallback
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import barrier, is_main_process, zero_first
from transformers.integrations.integration_utils import TensorBoardCallback
from transformers.utils import is_torch_bf16_gpu_available, is_torch_tf32_available

from checkpoint_utils import cleanup_checkpoints
from data_utils import find_all_jsonl_files
from mlfoundry_utils import (
    MLFoundryCallback,
    download_mlfoundry_artifact,
    get_or_create_run,
    is_mlfoundry_artifact,
    log_model_to_mlfoundry,
    sanitize_name,
)
from utils import (
    ExtraMetricsCallback,
    maybe_set_custom_tempdir,
    maybe_set_torch_max_memory,
    try_cleanup_gpus,
)

logger = logging.getLogger("axolotl")

# TODO:
# Save axolotl config when we create the callback
# Implement checkpoints fetching to resume

# Zero 3 loading fixes
# Support chat format data
# Support HF Hub Datasets
# Check if model fits in given gpus with TP to avoid and future crashes

# CURRENT LIMITATIONS
# Axolotl sets report_to to None instead of "none"
# There should be a data_seed
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


def _make_dataset_file_source(path, split="train"):
    return {
        "path": path,
        "ds_type": "json",
        "type": {
            "system_prompt": "",
            "field_system": "system",
            "field_instruction": "prompt",
            "field_output": "completion",
            "format": "{instruction} {input} ",
            "no_input_format": "{instruction}",
            "system_format": "{system}",
        },
        "split": split,
    }


def dataset_uri_to_axolotl_datasources(uri, download_dir):
    # TODO: Add support for HF datasets
    if uri.startswith("https://"):
        return [_make_dataset_file_source(path=uri)]
    elif is_mlfoundry_artifact(uri):
        datasources = []
        logger.info("Downloading artifact from mlfoundry")
        artifact_download_dir = os.path.join(download_dir, sanitize_name(uri))
        download_path = download_mlfoundry_artifact(
            artifact_version_fqn=uri, download_dir=artifact_download_dir, overwrite=True
        )
        for filepath in find_all_jsonl_files(download_path):
            logger.info("Adding jsonl file {filepath}")
            datasources.append(_make_dataset_file_source(path=filepath))
        return datasources
    elif os.path.exists(uri):
        datasources = []
        if os.path.isdir(uri):
            for filepath in find_all_jsonl_files(uri):
                datasources.append(_make_dataset_file_source(path=filepath))
        else:
            datasources = [_make_dataset_file_source(path=uri)]
        return datasources
    else:
        raise ValueError("Unsupported data uri or path does not exist: {uri}")


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

    data_dir = os.path.join(os.path.abspath(cfg.output_dir), "data")
    set_cfg_option_if_auto(cfg, "data_dir", data_dir)
    cfg.output_dir = os.path.join(os.path.abspath(cfg.output_dir), "model")
    axolotl_config = os.path.join(cfg.output_dir, "axolotl_config.yaml")

    if is_main_process():
        if cfg.cleanup_output_dir_on_start is True:
            logger.warning(f"--cleanup_output_dir_on_start was to set to True, wiping {cfg.output_dir}")
            shutil.rmtree(cfg.output_dir)

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
                set_cfg_option_if_auto(cfg, "mlfoundry_log_checkpoints", mlfoundry_checkpoint_artifact_name)
            else:
                cfg.mlfoundry_log_checkpoints = False
                cfg.mlfoundry_checkpoint_artifact_name = None

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
            cfg.datasets = dataset_uri_to_axolotl_datasources(uri=cfg.train_data_uri, download_dir=cfg.data_dir)
        if cfg.test_datasets == "auto":
            if cfg.eval_data_uri and str(cfg.eval_data_uri).lower() != "na":
                cfg.test_datasets = dataset_uri_to_axolotl_datasources(uri=cfg.eval_data_uri, download_dir=cfg.data_dir)
            elif cfg.val_set_size:
                set_cfg_option_if_auto(cfg, "test_datasets", [], force=True)
            else:
                raise ValueError(
                    "At least one of `eval_data_uri` or `val_set_size` must be non null when `test_datasets` is set to auto"
                )

        if cfg.test_datasets:
            set_cfg_option_if_auto(cfg, "val_set_size", 0, force=True)

            # TODO: Axolotl has a bug where it disables load_best_model_at_end when using explicit test datasets
            # https://github.com/OpenAccess-AI-Collective/axolotl/issues/1286
            # Remove this when resolved upstream
            set_cfg_option_if_auto(cfg, "early_stopping_patience", None, force=True)

        # TODO: Implement correct resuming
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
        with open(axolotl_config, "w") as f:
            yaml.dump(cfg, f)
    return axolotl_config


def patched_pretrain_hooks(cfg, trainer: AxolotlTrainer):
    # Bad hack because axolotl is not flexible at the moment
    if is_main_process():
        logger.info(f"Config: {cfg}")

    if is_main_process():
        mlfoundry_cb = None
        if cfg.mlfoundry_enable_reporting is True:
            run = get_or_create_run(
                ml_repo=cfg.mlfoundry_ml_repo,
                run_name=cfg.mlfoundry_run_name,
                auto_end=False,
                create_ml_repo=False,
            )
            mlfoundry_cb = MLFoundryCallback(
                run=run,
                log_checkpoints=cfg.mlfoundry_log_checkpoints,
                checkpoint_artifact_name=cfg.mlfoundry_checkpoint_artifact_name,
            )
        extra_metrics_cb = ExtraMetricsCallback()
        tensorboard_cb_idx = None
        for i, cb in enumerate(trainer.callback_handler.callbacks):
            if isinstance(cb, TensorBoardCallback):
                tensorboard_cb_idx = i
                break

        ax_gpu_stats_cb_idx = None
        for i, cb in enumerate(trainer.callback_handler.callbacks):
            if isinstance(cb, GPUStatsCallback):
                ax_gpu_stats_cb_idx = i
                break

        if tensorboard_cb_idx:
            # [..., TB_CB, ...]
            new_callbacks = [
                extra_metrics_cb,
                trainer.callback_handler.callbacks[tensorboard_cb_idx],
            ]
            if mlfoundry_cb:
                new_callbacks.append(mlfoundry_cb)
            trainer.callback_handler.callbacks[tensorboard_cb_idx : tensorboard_cb_idx + 1] = new_callbacks
            # [..., EM_CB, TB_CB, MLF_CB?, ...]
        elif ax_gpu_stats_cb_idx:
            # [..., AGS_CB, ...]
            new_callbacks = [
                extra_metrics_cb,
            ]
            if mlfoundry_cb:
                new_callbacks.append(mlfoundry_cb)
            trainer.callback_handler.callbacks[ax_gpu_stats_cb_idx:ax_gpu_stats_cb_idx] = new_callbacks
            # [..., EM_CB, MLF_CB?, AGS_CB, ...]
        else:
            logger.warning("Mlfoundry callback injection failed!")


def patched_post_train_hooks(cfg, trainer: AxolotlTrainer):
    if trainer.args.deepspeed and hasattr(trainer, "deepspeed") and hasattr(trainer.deepspeed, "destroy"):
        trainer.deepspeed.destroy()
    trainer.accelerator.free_memory()


def patch_train_hooks():
    import axolotl.train

    if hasattr(axolotl.train, "pretrain_hooks"):
        axolotl.train.pretrain_hooks = patched_pretrain_hooks
    else:
        raise ValueError(
            "Did not find `pretrain_hooks` on `axolotl.train`. " "This is required to patch and add callbacks"
        )

    if hasattr(axolotl.train, "post_train_hooks"):
        axolotl.train.post_train_hooks = patched_post_train_hooks
    else:
        raise ValueError(
            "Did not find `post_train_hooks` on `axolotl.train`. " "This is required for training to end correctly"
        )


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
    patch_train_hooks()
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
            metadata={},  # TODO
        )
        run.end()


if __name__ == "__main__":
    fire.Fire(train_with_truefoundry)


# @contextlib.contextmanager
# def deepspeed_zero3_disabled(training_arguments: HFTrainingArguments):
#     if training_arguments.deepspeed and is_deepspeed_zero3_enabled():
#         unset_hf_deepspeed_config()
#         yield
#         set_hf_deepspeed_config(training_arguments.hf_deepspeed_config)
#     else:
#         yield


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


# def dist_get_last_checkpoint_for_resume_if_any(
#     training_arguments: HFTrainingArguments,
#     other_arguments: OtherArguments,
# ):
#     last_checkpoint_dir = None
#     dist_s = training_arguments.distributed_state
#     last_checkpoint_info_path = os.path.join(CACHE_DIR, "last_checkpoint_info.json")
#     if dist_s.is_main_process:
#         last_checkpoint_dir = get_last_checkpoint_for_resume_if_any(
#             output_dir=training_arguments.output_dir,
#             resume_from_checkpoint=training_arguments.resume_from_checkpoint,
#             mlfoundry_enable_reporting=other_arguments.mlfoundry_enable_reporting,
#             mlfoundry_ml_repo=other_arguments.mlfoundry_ml_repo,
#             mlfoundry_checkpoint_artifact_name=other_arguments.mlfoundry_checkpoint_artifact_name,
#         )
#         with open(last_checkpoint_info_path, "w") as f:
#             last_checkpoint_info = {"last_checkpoint_dir": last_checkpoint_dir}
#             json.dump(last_checkpoint_info, f)
#     else:
#         with open(last_checkpoint_info_path, "r") as f:
#             last_checkpoint_info = json.load(f)
#         last_checkpoint_dir = last_checkpoint_info["last_checkpoint_dir"]

#     return last_checkpoint_dir
