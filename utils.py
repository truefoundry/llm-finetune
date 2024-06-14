import contextlib
import gc
import json
import logging
import math
import os
import tempfile
import time
from typing import Optional

import pynvml
import torch
from pydantic.v1 import BaseModel
from transformers import TrainerCallback

logger = logging.getLogger("axolotl")


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


def try_cleanup_gpus(n_iters=6, interval_seconds=10):
    for _ in range(n_iters):
        gc.collect()
        time.sleep(interval_seconds)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info(get_gpu_metrics())


def maybe_set_custom_tempdir():
    # We make sure any custom tempdir set by setting `TMPDIR` or equivalent env variables exist
    _tempdir = os.getenv("TMPDIR")
    if _tempdir:
        _tempdir = os.path.abspath(_tempdir)
        if os.path.exists(_tempdir) and os.path.isfile(_tempdir):
            raise ValueError("Current `TMPDIR` points to a file path, please set it to a directory path")
        else:
            os.makedirs(_tempdir, exist_ok=True)
        if tempfile.gettempdir() != _tempdir:
            tempfile.tempdir = _tempdir  # Not good, but necessary


def maybe_set_torch_max_memory(device: int):
    torch_per_process_memory_limit = os.getenv("TORCH_PER_PROCESS_MEMORY_LIMIT")
    if torch_per_process_memory_limit:
        if torch.cuda.is_available() and device >= 0:
            torch_per_process_memory_limit = float(torch_per_process_memory_limit)
            _, total = torch.cuda.mem_get_info()
            if torch_per_process_memory_limit <= 1.0:
                frac = torch_per_process_memory_limit
                torch_per_process_memory_limit = frac * total / (1024 * 1024)
            else:
                torch_per_process_memory_limit = int(torch_per_process_memory_limit)
                frac = (torch_per_process_memory_limit * 1024 * 1024) / total
            logger.info(f"Setting max memory limit on device {device} to {frac} ({torch_per_process_memory_limit} MiB)")
            torch.cuda.set_per_process_memory_fraction(frac, device=device)
    else:
        torch.cuda.set_per_process_memory_fraction(0.95, device=device)


@contextlib.contextmanager
def temporarily_unset_distributed_envs():
    old_envs = {}
    for key in os.environ:
        if key.startswith("ACCELERATE_") or key in {"WORLD_SIZE"}:
            old_envs[key] = os.environ.pop(key)
    yield
    os.environ.update(old_envs)


# Notebook Utils


class LaunchParameters(BaseModel):
    class Config:
        extra = "ignore"

    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_length: Optional[int] = 2048
    batch_size: int = 1


def load_launch_parameters(path):
    if os.path.exists(path):
        with open(path) as f:
            launch_parameters = LaunchParameters.parse_obj(json.load(f))
    else:
        launch_parameters = LaunchParameters()
        print(f"File `{path}` is missing, using defaults: {launch_parameters.dict()}")
    return launch_parameters
