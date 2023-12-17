import logging
import math

import pynvml
import torch
from transformers import TrainerCallback

logger = logging.getLogger("truefoundry-finetune")


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

    def _add_system_metrics(self, logs):
        gpu_count = torch.cuda.device_count()
        try:
            pynvml.nvmlInit()
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                utilz = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

                logs[f"system/gpu.{i}.utilization"] = utilz.gpu
                logs[f"system/gpu.{i}.memory_allocated"] = memory.used / (1024.0**2)
                logs[f"system/gpu.{i}.memory_allocated.percent"] = (memory.used / float(memory.total)) * 100
        except pynvml.NVMLError:
            pass

    # noinspection PyMethodOverriding
    def on_log(self, args, state, control, logs, model=None, **kwargs):
        # TODO (chiragjn): Hack for now, needs to be moved to `compute_metrics`
        #   unfortunately compute metrics does not give us already computed metrics like eval_loss
        if not state.is_world_process_zero:
            return

        self._add_perplexity(logs)
        self._add_system_metrics(logs)
        logger.info(f"Metrics: {logs}")
