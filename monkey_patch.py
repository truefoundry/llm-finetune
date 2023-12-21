import contextlib


def _deepspeed_load_checkpoint(deepspeed_engine, checkpoint_path):
    import glob

    from transformers.utils import is_peft_available

    deepspeed_checkpoint_dirs = sorted(glob.glob(f"{checkpoint_path}/global_step*"))

    if len(deepspeed_checkpoint_dirs) > 0:
        print(f"Attempting to resume from {checkpoint_path}")
        load_module_strict = True
        if is_peft_available():
            from peft import PeftModel

            if isinstance(deepspeed_engine.module, PeftModel):
                load_module_strict = False
        # this magically updates self.optimizer and self.lr_scheduler
        load_path, _ = deepspeed_engine.load_checkpoint(
            checkpoint_path,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
            load_module_strict=load_module_strict,
        )
        if load_path is None:
            raise ValueError(f"[deepspeed] failed to resume from checkpoint {checkpoint_path}")
    else:
        raise ValueError(f"Can't find a valid checkpoint at {checkpoint_path}")


@contextlib.contextmanager
def patched_deepspeed_load_checkpoint():
    import transformers.trainer

    old = transformers.trainer.deepspeed_load_checkpoint
    transformers.trainer.deepspeed_load_checkpoint = _deepspeed_load_checkpoint
    yield
    transformers.trainer.deepspeed_load_checkpoint = old
