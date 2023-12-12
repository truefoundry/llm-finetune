import contextlib
import logging

import torch.distributed

logger = logging.getLogger("truefoundry-finetune")


class DistributedState:
    def __init__(self, world_size: int, local_rank: int):
        self.world_size = world_size
        self.local_rank = local_rank

    @property
    def is_distributed(self):
        return self.world_size > 1

    @property
    def is_main_process(self):
        return self.local_rank <= 0

    @contextlib.contextmanager
    def main_process_first(self):
        if self.is_distributed:
            if not self.is_main_process:
                torch.distributed.barrier()
            yield
            if self.is_main_process:
                logger.info("Getting other ranks in sync with main process")
                torch.distributed.barrier()
        else:
            yield

    def wait_for_everyone(self):
        if self.is_distributed:
            torch.distributed.barrier()
