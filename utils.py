import json
import logging
import math
import os
from typing import Optional

import pynvml
import torch
from pydantic import BaseModel
from transformers import TrainerCallback

logger = logging.getLogger("truefoundry-finetune")


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


# Notebook Utils


class LaunchParameters(BaseModel):
    class Config:
        extra = "ignore"

    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_length: Optional[int] = None
    batch_size: int = 1


def load_launch_parameters(path):
    if os.path.exists(path):
        with open(path) as f:
            launch_parameters = LaunchParameters.parse_obj(json.load(f))
    else:
        launch_parameters = LaunchParameters()
        print(f"File `{path}` is missing, using defaults: {launch_parameters.dict()}")
    return launch_parameters
