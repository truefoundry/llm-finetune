import logging
import sys
from typing import Optional

from pydantic import ConfigDict
from transformers.integrations.integration_utils import TensorBoardCallback

from data_utils import DatasetType
from mlfoundry_utils import MLFoundryCallback, get_or_create_run
from utils import ExtraMetricsCallback

logger = logging.getLogger("axolotl")


def patched_validate_config(cfg, capabilities: Optional[dict] = None):
    from axolotl.utils.config.models.input.v0_4_1 import (
        AxolotlConfigWCapabilities,
        AxolotlInputConfig,
    )
    from axolotl.utils.dict import DictDefault

    class TruefoundryAxolotlInputConfig(AxolotlInputConfig):
        train_data_uri: Optional[str]
        val_data_uri: Optional[str] = None
        val_set_size: float = 0.1
        dataset_type: DatasetType = DatasetType.completion
        mlfoundry_enable_reporting: bool = False
        mlfoundry_ml_repo: Optional[str] = None
        mlfoundry_run_name: Optional[str] = None
        mlfoundry_log_checkpoints: bool = True
        mlfoundry_checkpoint_artifact_name: Optional[str] = None
        cleanup_output_dir_on_start: bool = False
        logging_dir: str = "./tensorboard_logs"

    class TruefoundryAxolotlConfigWCapabilities(AxolotlConfigWCapabilities, TruefoundryAxolotlInputConfig):
        model_config = ConfigDict(extra="allow")

    if capabilities:
        return DictDefault(
            dict(
                TruefoundryAxolotlConfigWCapabilities(**cfg.to_dict(), capabilities=capabilities).model_dump(
                    exclude_unset=True
                )
            )
        )
    return DictDefault(dict(TruefoundryAxolotlInputConfig(**cfg.to_dict()).model_dump(exclude_unset=True)))


def add_custom_prompt_strategies():
    import custom_prompt_strategies

    sys.modules["axolotl.prompt_strategies.custom_prompt_strategies"] = custom_prompt_strategies


def patched_pretrain_hooks(cfg, trainer):
    # type: (DictDefault, AxolotlTrainer) -> None
    # Bad hack because axolotl is not flexible at the moment
    from axolotl.utils.callbacks import GPUStatsCallback
    from axolotl.utils.distributed import is_main_process

    if not is_main_process():
        return
    logger.info(f"Config: {cfg}")
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
        raise Exception("Mlfoundry callback injection failed!")


def patched_post_train_hooks(cfg, trainer):
    # type: (DictDefault, AxolotlTrainer) -> None
    if trainer.args.deepspeed and hasattr(trainer, "deepspeed") and hasattr(trainer.deepspeed, "destroy"):
        trainer.deepspeed.destroy()
    trainer.accelerator.free_memory()


def monkey_patch_axolotl_internals():
    import axolotl.logging_config
    import axolotl.train
    import axolotl.utils.config

    axolotl.logging_config.DEFAULT_LOGGING_CONFIG["disable_existing_loggers"] = False
    axolotl.logging_config.configure_logging()

    if hasattr(axolotl.utils.config, "validate_config"):
        logger.info("Patching validate_config...")
        axolotl.utils.config.validate_config = patched_validate_config
    else:
        raise ValueError("Did not find `validate_config` on `axolotl.utils.config`. " "This is required")

    logger.info("Adding custom data prompt strategies...")
    add_custom_prompt_strategies()

    if hasattr(axolotl.train, "pretrain_hooks"):
        logger.info("Patching pretrain_hooks...")
        axolotl.train.pretrain_hooks = patched_pretrain_hooks
    else:
        raise ValueError(
            "Did not find `pretrain_hooks` on `axolotl.train`. " "This is required to patch and add callbacks"
        )

    if hasattr(axolotl.train, "post_train_hooks"):
        logger.info("Patching post_train_hooks...")
        axolotl.train.post_train_hooks = patched_post_train_hooks
    else:
        raise ValueError(
            "Did not find `post_train_hooks` on `axolotl.train`. " "This is required for training to end correctly"
        )
