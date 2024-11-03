import enum
import logging
import math
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pynvml
import torch
from axolotl.integrations.base import BasePlugin
from axolotl.utils.callbacks import GPUStatsCallback
from axolotl.utils.distributed import is_main_process
from pydantic import BaseModel, ConfigDict
from transformers import Trainer, TrainerCallback
from transformers.integrations import rewrite_logs
from transformers.integrations.integration_utils import TensorBoardCallback
from truefoundry import ml

if TYPE_CHECKING:
    from truefoundry.ml import MlFoundryRun

TFY_INTERNAL_JOB_NAME = os.getenv("TFY_INTERNAL_COMPONENT_NAME")
TFY_INTERNAL_JOB_RUN_NAME = os.getenv("TFY_INTERNAL_JOB_RUN_NAME")
logger = logging.getLogger("axolotl")


def _drop_non_finite_values(dct: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = {}
    for k, v in dct.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            if not math.isfinite(v):
                logger.warning(f"Dropping non-finite value for key={k} value={v!r}")
                continue
        sanitized[k] = v
    return sanitized


def get_gpu_metrics():
    gpu_count = torch.cuda.device_count()
    metrics = {}
    try:
        pynvml.nvmlInit()
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            utilz = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

            metrics[f"system/gpu.{i}.utilization"] = utilz.gpu
            metrics[f"system/gpu.{i}.memory_allocated"] = memory.used / (1024.0**2)
            metrics[f"system/gpu.{i}.memory_allocated.percent"] = (memory.used / float(memory.total)) * 100
    except pynvml.NVMLError:
        pass

    return metrics


def get_or_create_run(ml_repo: str, run_name: str, auto_end: bool = False):
    from truefoundry.ml.autogen.client.exceptions import ResourceDoesNotExist

    client = ml.get_client()
    try:
        run = client.get_run_by_name(ml_repo=ml_repo, run_name=run_name)
    except Exception as e:
        if not isinstance(e, ResourceDoesNotExist):
            raise
        run = client.create_run(ml_repo=ml_repo, run_name=run_name, auto_end=auto_end)
    return run


class ExtraMetricsCallback(TrainerCallback):
    def _add_perplexity(self, logs):
        for loss_key, perplexity_key in [
            ("loss", "train_perplexity"),
            ("eval_loss", "eval_perplexity"),
        ]:
            if loss_key in logs:
                try:
                    perplexity = math.exp(logs[loss_key])
                except OverflowError:
                    perplexity = float("inf")
                    logger.warning(f"Encountered inf in eval perplexity, cannot log it as a metric")
                logger.info(f"{perplexity_key}: {perplexity}")
                logs[perplexity_key] = perplexity

    # noinspection PyMethodOverriding
    def on_log(self, args, state, control, logs, model=None, **kwargs):
        # TODO (chiragjn): Hack for now, needs to be moved to `compute_metrics`
        #   unfortunately compute metrics does not give us already computed metrics like eval_loss
        if not state.is_world_process_zero:
            return

        self._add_perplexity(logs)
        logs.update(get_gpu_metrics())
        logger.info(f"Metrics: {logs}")


class TrueFoundryMLCallback(TrainerCallback):
    def __init__(
        self,
        run: "MlFoundryRun",
        log_checkpoints: bool = True,
        checkpoint_artifact_name: Optional[str] = None,
    ):
        self._run = run
        self._checkpoint_artifact_name = checkpoint_artifact_name
        self._log_checkpoints = log_checkpoints

        if not self._checkpoint_artifact_name:
            logger.warning("checkpoint_artifact_name not passed. Checkpoints will not be logged to MLFoundry")

    # noinspection PyMethodOverriding
    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if not state.is_world_process_zero:
            return

        metrics = {}
        for k, v in logs.items():
            if k.startswith("system/gpu"):
                continue
            if isinstance(v, (int, float, np.integer, np.floating)) and math.isfinite(v):
                metrics[k] = v
            else:
                logger.warning(
                    f'Trainer is attempting to log a value of "{v}" of'
                    f' type {type(v)} for key "{k}" as a metric.'
                    " Mlfoundry's log_metric() only accepts finite float and"
                    " int types so we dropped this attribute."
                )
        self._run.log_metrics(rewrite_logs(metrics), step=state.global_step)

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        if not self._log_checkpoints:
            return

        if not self._checkpoint_artifact_name:
            return

        ckpt_dir = f"checkpoint-{state.global_step}"
        artifact_path = os.path.join(args.output_dir, ckpt_dir)
        description = None
        if TFY_INTERNAL_JOB_NAME:
            description = f"Checkpoint from finetuning job={TFY_INTERNAL_JOB_NAME} run={TFY_INTERNAL_JOB_RUN_NAME}"
        logger.info(f"Uploading checkpoint {ckpt_dir} ...")
        metadata = {}
        for log in state.log_history:
            if isinstance(log, dict) and log.get("step") == state.global_step:
                metadata = log.copy()
        metadata = _drop_non_finite_values(metadata)
        self._run.log_artifact(
            name=self._checkpoint_artifact_name,
            artifact_paths=[(artifact_path,)],
            metadata=metadata,
            step=state.global_step,
            description=description,
        )


class DatasetType(str, enum.Enum):
    completion = "completion"
    chat = "chat"


class LongSequenceStrategy(str, enum.Enum):
    error = "error"
    drop = "drop"
    truncate = "truncate"


class TruefoundryMLPluginArgs(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    dataset_type: DatasetType = DatasetType.chat
    train_data_uri: Optional[str]
    val_data_uri: Optional[str] = None
    val_set_size: float = 0.1

    long_sequences_strategy: LongSequenceStrategy = LongSequenceStrategy.error

    truefoundry_ml_enable_reporting: bool = False
    truefoundry_ml_repo: Optional[str] = None
    truefoundry_ml_run_name: Optional[str] = None
    truefoundry_ml_log_checkpoints: bool = True
    truefoundry_ml_checkpoint_artifact_name: Optional[str] = None

    cleanup_output_dir_on_start: bool = False
    logging_dir: str = "./tensorboard_logs"


class TrueFoundryMLPlugin(BasePlugin):
    def get_input_args(self):
        return "axolotl_truefoundry.TruefoundryMLPluginArgs"

    def add_callbacks_post_trainer(self, cfg: TruefoundryMLPluginArgs, trainer: Trainer) -> List[TrainerCallback]:
        # Note: `cfg` is not really an instance of `TruefoundryMLPluginArgs` but a `DictDefault` object
        if not is_main_process():
            return []
        logger.info(f"Config: {cfg}")
        truefoundry_ml_cb = None
        if cfg.truefoundry_ml_enable_reporting is True:
            run = get_or_create_run(
                ml_repo=cfg.truefoundry_ml_repo,
                run_name=cfg.truefoundry_ml_run_name,
                auto_end=False,
            )
            truefoundry_ml_cb = TrueFoundryMLCallback(
                run=run,
                log_checkpoints=cfg.truefoundry_ml_log_checkpoints,
                checkpoint_artifact_name=cfg.truefoundry_ml_checkpoint_artifact_name,
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
            if truefoundry_ml_cb:
                new_callbacks.append(truefoundry_ml_cb)
            trainer.callback_handler.callbacks[tensorboard_cb_idx : tensorboard_cb_idx + 1] = new_callbacks
            # [..., EM_CB, TB_CB, MLF_CB?, ...]
        elif ax_gpu_stats_cb_idx:
            # [..., AGS_CB, ...]
            new_callbacks = [
                extra_metrics_cb,
            ]
            if truefoundry_ml_cb:
                new_callbacks.append(truefoundry_ml_cb)
            trainer.callback_handler.callbacks[ax_gpu_stats_cb_idx:ax_gpu_stats_cb_idx] = new_callbacks
            # [..., EM_CB, MLF_CB?, AGS_CB, ...]
        else:
            raise Exception("TrueFoundry ML callback injection failed!")

        return []
