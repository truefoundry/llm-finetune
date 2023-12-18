import contextlib
import gc
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import bitsandbytes as bnb
import mlfoundry
import torch
from accelerate import infer_auto_device_map, init_empty_weights
from datasets import DatasetDict
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    HfArgumentParser,
    IntervalStrategy,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
    set_hf_deepspeed_config,
    unset_hf_deepspeed_config,
)
from transformers.integrations.integration_utils import TensorBoardCallback
from transformers.training_args import ParallelMode
from transformers.utils import is_torch_tf32_available
from transformers.utils import logging as hf_logging_utils

from checkpoint_utils import cleanup_checkpoints, get_last_checkpoint_for_resume_if_any
from data_utils import SequenceDataCollator, build_dataset, get_data
from mlfoundry_utils import (
    MLFoundryCallback,
    get_or_create_run,
    log_model_to_mlfoundry,
    sanitize_name,
)
from utils import ExtraMetricsCallback

# TODO (chiragjn):
#   - Test deepspeed with resume
#   - Test and fix Deepspeed (Zero 3) weight gathering bugs during checkpointing if any
#   - Invent some work around to use auto batch size finder with deepspeed
#   - Add support for dataset packing
#   - Add support for dataset streaming
#   - Add support to read dataset from HF
#   - Add support to push to HF Hub

DEBUG = (os.getenv("DEBUG") or "").lower() == "true"
TFY_INTERNAL_JOB_NAME = os.getenv("TFY_INTERNAL_COMPONENT_NAME")
TFY_INTERNAL_JOB_RUN_NAME = os.getenv("TFY_INTERNAL_JOB_RUN_NAME")
THIS_DIR = os.path.abspath(os.path.dirname(__name__))
CACHE_DIR = os.path.join(THIS_DIR, ".cache")
logger = logging.getLogger("truefoundry-finetune")


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_KEY = "prompt"
COMPLETION_KEY = "completion"


@dataclass
class HFTrainingArguments(TrainingArguments):
    def __post_init__(self):
        self.tf32 = not self.use_cpu and torch.cuda.is_available() and is_torch_tf32_available()
        if self.save_strategy == IntervalStrategy.NO:
            self.load_best_model_at_end = False
        _resume = self.resume_from_checkpoint
        if _resume and _resume.strip().lower() in ("true", "false"):
            self.resume_from_checkpoint = _resume.strip().lower() == "true"
        if self.gradient_checkpointing:
            self.gradient_checkpointing_kwargs = self.gradient_checkpointing_kwargs or {}
            self.gradient_checkpointing_kwargs["use_reentrant"] = False
        if self.auto_find_batch_size and self.deepspeed:
            raise ValueError(
                f"Auto batch size finder is not supported with Deepseed because of bugs with model preparation: https://github.com/huggingface/transformers/issues/24558"
            )
        super().__post_init__()


@dataclass
class OtherArguments:
    model_id: str = field(metadata={"help": "Huggingface hub model ID"})
    train_data: str = field(metadata={"help": "URL to the jsonl training dataset"})
    eval_size: Optional[float] = field(
        default=0.1,
        metadata={"help": "Proportion of training data to use as evaluation set. Ignored if `eval_data` is passed"},
    )
    eval_data: Optional[str] = field(
        default="NA",
        metadata={"help": "URL to the jsonl evaluation dataset. Overrides eval_size. Leave as NA if not available"},
    )
    train_on_prompt: bool = field(
        default=False,
        metadata={"help": "If to train on prompt and include it in the loss"},
    )
    pad_to_multiple_of: int = field(
        default=64,
        metadata={"help": "Pad the sequences batch to multiple of this"},
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "If to use flash attention to speed up training - only supported on some architectures!"},
    )
    use_ddp: bool = field(
        default=False,
        metadata={"help": "If to use DDP - only applicable when multiple gpus are available"},
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "If to train the model with LoRa"},
    )
    lora_r: int = field(
        default=32,
        metadata={"help": "r value for lora config"},
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "value of alpha for lora config"},
    )
    lora_target_modules: str = field(
        default="auto",
        metadata={"help": "The names of the modules to apply Lora to"},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "The dropout probability for Lora layers."},
    )
    lora_bias: str = field(
        default="none",
        metadata={
            "help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'. If 'all' or 'lora_only', the corresponding biases will be updated during training."
        },
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "If to train the model with qLoRa"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "quantization data type options are {'nf4', 'fp4'}, by default it is nf4"},
    )
    use_double_quant: bool = field(
        default=True,
        metadata={
            "help": "This flag is used for nested quantization where the quantization constants from the first quantization are quantized again"
        },
    )
    qlora_bit_length: int = field(
        default=4,
        metadata={"help": "To enable 8 bit quantization set this to 8 or else by default it is 4"},
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Max length to truncate the examples to. By default we try to pick "
            "from tokenizer config (default: None)"
        },
    )
    max_num_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For quick debugging purposes, how many samples to use (default: all)"},
    )
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={
            "help": "How many evaluation steps to wait for the metric (loss) to improve before stopping. None or 0 means early stopping is not applied (default: None)"
        },
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={
            "help": "When early stopping is enabled the metric (loss) should improve by at least this much to not count towards patience/stop (default: 0.0)"
        },
    )
    mlfoundry_enable_reporting: bool = field(
        default=True,
        metadata={"help": "Use mlfoundry to log metrics, checkpoints and model"},
    )
    mlfoundry_ml_repo: Optional[str] = field(default=None, metadata={"help": "ML Repo to put the model to"})
    mlfoundry_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Run name for mlfoundry run"},
    )
    mlfoundry_log_checkpoints: bool = field(
        default=True,
        metadata={"help": "If to log intermediate checkpoints to mlfoundry"},
    )
    mlfoundry_checkpoint_artifact_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "ML Repo artifact name to save checkpoints. \n"
            "The artifact will be created if it does not exist under the give ML Repo"
        },
    )
    cleanup_output_dir_on_start: bool = field(
        default=False,
        metadata={"help": "Cleanup output dir at the start of training run"},
    )


class HFTrainer(Trainer):
    def _wrap_model(self, model, training=True, dataloader=None):
        outputs = super()._wrap_model(model=model, training=training, dataloader=dataloader)
        if (
            self.args.parallel_mode == ParallelMode.DISTRIBUTED
            and self.accelerator.ddp_handler
            and not self.args.deepspeed
        ):
            self.accelerator.ddp_handler.gradient_as_bucket_view = True

        return outputs

    # def _inner_training_loop(
    #     self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    # ):
    #     # Hack to get around: https://github.com/huggingface/transformers/issues/24558
    #     # Note: Use this with caution! If the GPU memory allocation is uneven then ranks might not agree on the same batch size and things hang
    #     dist_s = self.args.distributed_state
    #     if self.args.auto_find_batch_size and self.args.deepspeed:
    #         self.model_wrapped = self.model
    #         self.deepspeed = None
    #         self.accelerator.free_memory()
    #         if torch.cuda.is_available():
    #             torch.cuda.synchronize()
    #         dist_s.wait_for_everyone()
    #     return super()._inner_training_loop(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)

    # TODO: incomplete! To address the above issue where ranks can disagree on batch size, one way is we communicate across all ranks and sync the batch size forcefully
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     from accelerate.utils.memory import should_reduce_batch_size
    #     exc = None
    #     dist_s = self.args.distributed_state

    #     try:
    #         outputs = super().compute_loss(model, inputs, return_outputs)
    #     except Exception as re:
    #         if should_reduce_batch_size(re):
    #             logger.info(f"CUDA OOM when trying {self._train_batch_size}. Crashing this rank!")
    #             exc = re
    #             # TODO: Inform other ranks to also crash if they haven't
    #     dist_s.wait_for_everyone()
    #     if not exc:
    #         logger.info(f"Forward went fine with {self._train_batch_size}. Checking in with other ranks")
    #         # TODO: Check other ranks if any of them has crashed and set exc is yes
    #     dist_s.wait_for_everyone()
    #     if exc:
    #         raise exc

    #     return outputs


def get_torch_dtype(training_arguments: HFTrainingArguments):
    torch_dtype = None
    if training_arguments.bf16:
        torch_dtype = torch.bfloat16
    elif training_arguments.fp16:
        torch_dtype = torch.float16
    return torch_dtype


def _cleanup_gpus():
    # TODO (chiragjn): We do not want anything to be offloaded to cpu or disk otherwise merging adapter fails!
    # This is a known issue with fix in progress
    #   - https://github.com/huggingface/peft/pull/1063
    #   - https://github.com/huggingface/transformers/pull/27412
    # Yes, sleeping is stupid but necessary till the above PRs are merged and made available in a new version
    for _ in range(6):
        gc.collect()
        time.sleep(10)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def merge_adapters_if_any(training_arguments: HFTrainingArguments, other_arguments: OtherArguments):
    check_if_model_will_fit_only_with_gpus(training_arguments=training_arguments, other_arguments=other_arguments)
    logger.info("Loading model and lora layers for merging ...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        training_arguments.output_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=get_torch_dtype(training_arguments),
        device_map="balanced",
    )
    logger.info("Merging lora adapter into main model. This can take a while ...")
    model = model.merge_and_unload()
    model.save_pretrained(training_arguments.output_dir, safe_serialization=True)
    for filename in [
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
    ]:
        file_to_delete = os.path.join(training_arguments.output_dir, filename)
        if os.path.exists(file_to_delete):
            os.remove(file_to_delete)


def filter_trainer_args_for_logging(
    training_arguments: TrainingArguments, other_arguments: OtherArguments
) -> Dict[str, Any]:
    # TODO (chiragjn): Update this list
    arguments = {
        "num_train_epochs": training_arguments.num_train_epochs,
        "per_device_train_batch_size": training_arguments.per_device_train_batch_size,
        "learning_rate": training_arguments.learning_rate,
        "lr_scheduler_type": training_arguments.lr_scheduler_type,
        "weight_decay": training_arguments.weight_decay,
        "max_grad_norm": training_arguments.max_grad_norm,
        "gradient_accumulation_steps": training_arguments.gradient_accumulation_steps,
        "warmup_ratio": training_arguments.warmup_ratio,
        "use_lora": other_arguments.use_lora,
        "use_qlora": other_arguments.use_qlora,
    }
    if other_arguments.use_lora or other_arguments.use_qlora:
        lora_args = {
            "lora_r": other_arguments.lora_r,
            "lora_alpha": other_arguments.lora_alpha,
            "lora_target_modules": other_arguments.lora_target_modules,
            "lora_dropout": other_arguments.lora_dropout,
            "lora_bias": other_arguments.lora_bias,
        }
        arguments.update(lora_args)

    if other_arguments.use_qlora:
        qlora_args = {
            "qlora_bit_length": other_arguments.qlora_bit_length,
            "bnb_4bit_quant_type": other_arguments.bnb_4bit_quant_type,
            "use_double_quant": other_arguments.use_double_quant,
        }
        arguments.update(qlora_args)

    return arguments


def _log_model_parameters(model):
    global DEBUG
    if not DEBUG:
        return
    logger.info("=== Model Parameters ===")
    for name, parameter in model.named_parameters():
        logger.info(f"\t {name}, {parameter.dtype}, {parameter.device}, {parameter.requires_grad}")
    logger.info("========================")


def _maybe_set_custom_tempdir():
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


def _maybe_set_torch_max_memory(device: int):
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


def _setup_logging(training_arguments: HFTrainingArguments):
    global logger

    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)
            hf_logging_utils.remove_handler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt=f"%(asctime)s [Rank-{training_arguments.local_rank}] %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    hf_logging_utils.disable_default_handler()
    hf_logging_utils.add_handler(handler)


def setup(training_arguments: HFTrainingArguments, other_arguments: OtherArguments):
    os.makedirs(CACHE_DIR, exist_ok=True)
    _setup_logging(training_arguments=training_arguments)
    _maybe_set_custom_tempdir()
    _maybe_set_torch_max_memory(device=training_arguments.local_rank)

    if other_arguments.use_flash_attention:
        import flash_attn as _


def find_all_linear_names(model, other_arguments: OtherArguments, exclude_lm_head: bool = True):
    lora_module_names = set()
    target_cls_type = torch.nn.Linear
    if other_arguments.use_qlora and other_arguments.qlora_bit_length == 8:
        target_cls_type = bnb.nn.Linear8bitLt
    elif other_arguments.use_qlora and other_arguments.qlora_bit_length == 4:
        target_cls_type = bnb.nn.Linear4bit
    for name, module in model.named_modules():
        if isinstance(module, target_cls_type):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if exclude_lm_head and "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    if len(list(lora_module_names)) == 0:
        raise ValueError(
            "Cannot automatically find target modules for LoRa please provide --lora_target_modules explicitly"
        )
    return list(lora_module_names)


def get_model(
    model_source: str,
    model_config,
    training_arguments: HFTrainingArguments,
    other_arguments: OtherArguments,
    device_map=None,
):
    dist_s = training_arguments.distributed_state
    logger.info("Loading model...")
    model_load_kwargs = {
        "low_cpu_mem_usage": True,
        "use_cache": False if training_arguments.gradient_checkpointing else True,
    }
    if model_config.architectures and model_config.architectures[0] == "PhiForCausalLM":
        model_load_kwargs.pop("use_cache", None)

    if other_arguments.use_flash_attention:
        model_load_kwargs["attn_implementation"] = "flash_attention_2"

    if other_arguments.use_qlora:
        compute_dtype = get_torch_dtype(training_arguments)
        torch_dtype = torch.float32
        if training_arguments.bf16:
            torch_dtype = torch.bfloat16
        elif training_arguments.fp16:
            # https://github.com/artidoro/qlora/blob/7f4e95a68dc076bea9b3a413d2b512eca6d004e5/qlora.py#L327C9-L327C104
            # QLoRA authors report instability when using float16
            torch_dtype = torch.float32
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=other_arguments.qlora_bit_length == 4,
            load_in_8bit=other_arguments.qlora_bit_length == 8,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=other_arguments.use_double_quant,
            bnb_4bit_quant_type=other_arguments.bnb_4bit_quant_type,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            quantization_config=bnb_config,
            device_map=device_map,
            **model_load_kwargs,
        )
        if dist_s.is_main_process:
            _log_model_parameters(model)
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_arguments.gradient_checkpointing
        )
        if dist_s.is_main_process:
            _log_model_parameters(model)
        # TODO (chiragjn): This is disabled because resuming does not work: https://github.com/TimDettmers/bitsandbytes/issues/782
        # training_arguments.optim = "paged_adamw_32bit"
    else:
        if training_arguments.deepspeed and is_deepspeed_zero3_enabled:
            model_load_kwargs.pop("low_cpu_mem_usage", None)
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            trust_remote_code=True,
            torch_dtype=get_torch_dtype(training_arguments),
            device_map=device_map,
            **model_load_kwargs,
        )

    if training_arguments.gradient_checkpointing:
        model.config.use_cache = False
    return model


def get_peft_wrapped_model(
    model,
    training_arguments: HFTrainingArguments,
    other_arguments: OtherArguments,
    _device_map=None,
    _checkpoint_dir: Optional[str] = None,
):
    dist_s = training_arguments.distributed_state
    # if _checkpoint_dir:
    #     model = PeftModel.from_pretrained(
    #         model=model,
    #         model_id=checkpoint_dir,
    #         is_trainable=True,
    #         device_map=_device_map
    #     )
    # else:
    if other_arguments.lora_target_modules == "auto":
        modules = find_all_linear_names(model, exclude_lm_head=True, other_arguments=other_arguments)
    else:
        modules = json.loads(other_arguments.lora_target_modules)
    logger.info(f"Modules targeted for lora are {modules}")

    other_arguments.lora_config = LoraConfig(
        **dict(
            r=other_arguments.lora_r,
            lora_alpha=other_arguments.lora_alpha,
            target_modules=modules,
            lora_dropout=other_arguments.lora_dropout,
            bias=other_arguments.lora_bias,
            task_type="CAUSAL_LM",
        )
    )
    logger.info("Applying peft config ...")
    other_arguments.lora_config.inference_mode = False
    model = get_peft_model(model, other_arguments.lora_config)

    # Taken from https://github.com/artidoro/qlora/blob/7f4e95a68dc076bea9b3a413d2b512eca6d004e5/qlora.py#L396
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_arguments.bf16:
                module = module.to(torch.bfloat16)
            if "norm" in name:
                # TODO (chiragjn): This is no longer always required. For e.g. LlamaRMSProp handles half precision correctly
                # but right now even prepare_model_for_k_bit does it
                module = module.to(torch.float32)
        if any(
            ename in name
            for ename in (
                "lm_head",
                "embed_tokens",
                "embed_in",
                "embed_out",
                "wte",
                "wpe",
            )
        ):
            if hasattr(module, "weight"):
                if training_arguments.bf16 and module.weight.dtype == torch.float32:
                    # This is experimental, normally qlora repo uses it but some others don't recommend it. So far we haven't see major problems.
                    # Note this downcasting is not recommended when using float16, that can cause numerical instability
                    module = module.to(torch.bfloat16)

    model.enable_input_require_grads()
    model.print_trainable_parameters()
    if dist_s.is_main_process:
        _log_model_parameters(model)
    return model


def get_tokenizer(model_source: str):
    logger.info("Loading tokenizer...")
    try:
        # Note: First we try loading with use_fast=False because for some models conversion takes too long
        tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, use_fast=False)
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=True,
        )
    logger.info(f"Tokenizer's padding side is {tokenizer.padding_side}")
    # There are some strategies that also assign unk token as pad token
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        logger.info("Pad token missing, adding a pad token")
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        logger.info("EOS token missing, adding a EOS token")
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        logger.info("BOS token missing, adding a BOS token")
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        logger.info("UNK token missing, adding a UNK token")
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    # TODO (chiragjn): Consider adding fake tokens to vocab to pad to multiple of 64. Can provide better throughput
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer, num_new_tokens


def get_max_length(max_length, tokenizer, model_config):
    logger.info("Resolving max_length for truncation...")
    if max_length is None:
        if tokenizer.model_max_length > int(1e6):
            logger.info(f"tokenizer config does not have proper model_max_length set. Looking at model config")
            for length_setting in [
                "max_sequence_length",
                "n_positions",
                "max_position_embeddings",
            ]:
                max_length = getattr(model_config, length_setting, None)
                if max_length:
                    logger.info(f"Assuming value of {length_setting} from model config as max length: {max_length}")
                    break
            if not max_length:
                logger.info(f"Found no max length setting, falling back to default of 512")
                max_length = 512
        else:
            max_length = tokenizer.model_max_length
    logger.info(f"Finally using max_length: {max_length}")
    return max_length


@contextlib.contextmanager
def deepspeed_zero3_disabled(training_arguments: HFTrainingArguments):
    if training_arguments.deepspeed and is_deepspeed_zero3_enabled():
        unset_hf_deepspeed_config()
        yield
        set_hf_deepspeed_config(training_arguments.hf_deepspeed_config)
    else:
        yield


def check_if_model_will_fit_only_with_gpus(
    training_arguments: HFTrainingArguments,
    other_arguments: OtherArguments,
):
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            other_arguments.model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=get_torch_dtype(training_arguments),
        )
    device_map = infer_auto_device_map(model, dtype=get_torch_dtype(training_arguments))
    logger.info(f"Inferred device_map for auto settings: {device_map}")
    if any(not isinstance(v, int) for v in device_map.values()):
        raise RuntimeError(
            "For lora/qlora the model must entirely fit on gpus without any kind of offloading to prevent bugs with merging! "
            "With the current configuration model is being offloaded to cpu/disk. This causes incorrect model saving. See https://github.com/huggingface/peft/issues/868"
        )


def dist_get_last_checkpoint_for_resume_if_any(
    training_arguments: HFTrainingArguments,
    other_arguments: OtherArguments,
):
    last_checkpoint_dir = None
    dist_s = training_arguments.distributed_state
    last_checkpoint_info_path = os.path.join(CACHE_DIR, "last_checkpoint_info.json")
    if dist_s.is_main_process:
        last_checkpoint_dir = get_last_checkpoint_for_resume_if_any(
            output_dir=training_arguments.output_dir,
            resume_from_checkpoint=training_arguments.resume_from_checkpoint,
            mlfoundry_enable_reporting=other_arguments.mlfoundry_enable_reporting,
            mlfoundry_ml_repo=other_arguments.mlfoundry_ml_repo,
            mlfoundry_checkpoint_artifact_name=other_arguments.mlfoundry_checkpoint_artifact_name,
        )
        with open(last_checkpoint_info_path, "w") as f:
            last_checkpoint_info = {"last_checkpoint_dir": last_checkpoint_dir}
            json.dump(last_checkpoint_info, f)
    else:
        with open(last_checkpoint_info_path, "r") as f:
            last_checkpoint_info = json.load(f)
        last_checkpoint_dir = last_checkpoint_info["last_checkpoint_dir"]

    return last_checkpoint_dir


def dist_build_dataset(
    train_data,
    eval_data,
    tokenizer,
    max_length,
    train_on_prompt: bool,
    training_arguments: HFTrainingArguments,
):
    dist_s = training_arguments.distributed_state
    logger.info("Building dataset...")
    dataset_cache_path = os.path.join(CACHE_DIR, "dataset")
    if dist_s.is_main_process:
        dataset_dict = build_dataset(
            train_data=train_data,
            eval_data=eval_data,
            tokenizer=tokenizer,
            max_length=max_length,
            train_on_prompt=train_on_prompt,
        )
        dataset_dict.save_to_disk(dataset_cache_path)
    else:
        logger.info("Loading datasets from cache ...")
        dataset_dict = DatasetDict.load_from_disk(dataset_cache_path)
    dataset_dict = dataset_dict.with_format("torch")
    train_dataset, eval_dataset = dataset_dict["train"], dataset_dict["eval"]
    logger.info(f"Train data size: {len(train_dataset)}")
    logger.info(f"Eval data size: {len(eval_dataset)}")
    return train_dataset, eval_dataset


def _train(
    *,
    training_arguments: HFTrainingArguments,
    other_arguments: OtherArguments,
    run: Optional[mlfoundry.MlFoundryRun] = None,
):
    dist_s = training_arguments.distributed_state
    dist_s.wait_for_everyone()
    set_seed(training_arguments.seed)

    if not dist_s.is_main_process:
        logger.info("Waiting for main process to load data, process it and fetch any checkpoints ...")

    with dist_s.main_process_first():
        if dist_s.is_main_process:
            train_data, eval_data = get_data(
                train_data=other_arguments.train_data,
                eval_data=other_arguments.eval_data,
                eval_size=other_arguments.eval_size,
                max_num_samples=other_arguments.max_num_samples,
                data_seed=training_arguments.data_seed,
            )
        else:
            train_data, eval_data = None, None

        last_checkpoint_dir = dist_get_last_checkpoint_for_resume_if_any(
            training_arguments=training_arguments, other_arguments=other_arguments
        )

        logger.info("Loading config ...")
        model_config = AutoConfig.from_pretrained(other_arguments.model_id, trust_remote_code=True)

        if last_checkpoint_dir:
            model_source = last_checkpoint_dir
        else:
            model_source = other_arguments.model_id

        tokenizer, num_new_tokens = get_tokenizer(model_source)

        max_length = get_max_length(
            max_length=other_arguments.max_length,
            tokenizer=tokenizer,
            model_config=model_config,
        )

        train_dataset, eval_dataset = dist_build_dataset(
            train_data=train_data,
            eval_data=eval_data,
            tokenizer=tokenizer,
            max_length=max_length,
            train_on_prompt=other_arguments.train_on_prompt,
            training_arguments=training_arguments,
        )

    no_of_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if training_arguments.deepspeed:
        device_map = None
    elif other_arguments.use_ddp:
        if no_of_gpus > 1:
            device_map = {"": "cuda:" + str(training_arguments.local_rank)}
        else:
            device_map = "auto"
    else:
        device_map = "auto"

    # TODO (chiragjn): Ideally we should be loading from checkpoint when available because we resize embeddings in some cases
    #   but because of device movement bugs with peft we are loading the pretrained model and re-applying peft config
    #   Details:
    #   Currently AutoModelForCausalLM.from_pretrained can load adapters from checkpoint but it does not return an instance of PeftModel,
    #   but some training code relies on checking if instance is PeftModel type
    #   so we want to re-load the model as PeftModel instance using the above PeftModel.from_pretrained
    #   Now the problem is when we load using AutoModelForCausalLM from a checkpoint it already has the lora layers in it
    #   so PeftModel.from_pretrained updates the lora layers and re-inits them
    #   (re-init is not a problem because resume_from_checkpoint in trainer should restore weights)
    #   However the layer updating code is broken that it does not move the layers to correct device.
    #   So you'll get base layers on gpu and lora layers on cpu crashing the code.
    #   There is a massive refactor in peft 0.7.0 which has mostly solved this but will need some time to migrate correctly
    #   So for now, we always load the base model from pretrained version, resize embeddings is tokenizer from checkpoint has more tokens, and re-apply the peft config from scratch
    model = get_model(
        model_source=other_arguments.model_id,  # This is not a bug
        model_config=model_config,
        device_map=device_map,
        training_arguments=training_arguments,
        other_arguments=other_arguments,
    )
    # TODO (chiragjn): If there are new tokens added, check if we want grads to be enabled on embedding and lm head.
    #   prepare_model_for_k_bit actually disables grad on embedding and lm head
    if model.get_input_embeddings().num_embeddings < len(tokenizer):
        logger.info(
            f"Resizing embeddings layer for newly added tokens. "
            f"Tokenizer length is {len(tokenizer)} but model embedding "
            f"layer has {model.get_input_embeddings().num_embeddings}"
        )
        model.resize_token_embeddings(len(tokenizer))

    if other_arguments.use_lora or other_arguments.use_qlora:
        model = get_peft_wrapped_model(
            model,
            training_arguments=training_arguments,
            other_arguments=other_arguments,
        )
    logger.info("Training...")
    # TODO (chiragjn): Add text generation metrics to `compute_metrics
    callbacks = [ExtraMetricsCallback(), TensorBoardCallback()]
    if run:
        callbacks.append(
            MLFoundryCallback(
                run=run,
                log_checkpoints=other_arguments.mlfoundry_log_checkpoints,
                checkpoint_artifact_name=other_arguments.mlfoundry_checkpoint_artifact_name,
            )
        )
    if other_arguments.early_stopping_patience:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=other_arguments.early_stopping_patience,
                early_stopping_threshold=other_arguments.early_stopping_threshold,
            )
        )
    trainer = HFTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_arguments,
        data_collator=SequenceDataCollator(tokenizer=tokenizer, multiple_of=other_arguments.pad_to_multiple_of),
        callbacks=callbacks,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    dist_s.wait_for_everyone()

    trainer.train(resume_from_checkpoint=last_checkpoint_dir)

    dist_s.wait_for_everyone()

    logger.info("Saving model...")
    if dist_s.is_main_process:
        cleanup_checkpoints(output_dir=training_arguments.output_dir)

    dist_s.wait_for_everyone()

    trainer.save_model(output_dir=training_arguments.output_dir)

    dist_s.wait_for_everyone()


def train(training_arguments: HFTrainingArguments, other_arguments: OtherArguments):
    dist_s = training_arguments.distributed_state
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    logger.info(f"Training Arguments: {training_arguments}")
    logger.info(f"Arguments: {other_arguments}")

    if other_arguments.use_qlora:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("No GPUs detected. We need at least one gpu available for Lora/QLora finetuning!")
        if training_arguments.deepspeed and is_deepspeed_zero3_enabled():
            raise ValueError("Deepspeed Zero 3 is currently not supported with QLoRa")

    if other_arguments.use_flash_attention and not (training_arguments.bf16 or training_arguments.fp16):
        raise ValueError("--use_flash_attention requires either --bf16 or --fp16")

    setup(training_arguments=training_arguments, other_arguments=other_arguments)

    run = None
    if dist_s.is_main_process and other_arguments.mlfoundry_enable_reporting:
        if not other_arguments.mlfoundry_run_name:
            if TFY_INTERNAL_JOB_RUN_NAME:
                fallback_run_name = f"finetune-{sanitize_name(TFY_INTERNAL_JOB_RUN_NAME)}"
            else:
                fallback_run_name = f"finetune-{timestamp}"
            logger.info(f"Setting --mlfoundry_run_name automatically to {fallback_run_name}")
            other_arguments.mlfoundry_run_name = fallback_run_name

        run = get_or_create_run(
            ml_repo=other_arguments.mlfoundry_ml_repo,
            run_name=other_arguments.mlfoundry_run_name,
            auto_end=False,
            create_ml_repo=False,
        )
        if other_arguments.mlfoundry_log_checkpoints:
            if not other_arguments.mlfoundry_checkpoint_artifact_name:
                if TFY_INTERNAL_JOB_RUN_NAME:
                    mlfoundry_checkpoint_artifact_name = f"ckpt-{sanitize_name(TFY_INTERNAL_JOB_RUN_NAME)}"
                else:
                    mlfoundry_checkpoint_artifact_name = f"ckpt-{run.run_name}"
                logger.info(
                    f"Setting --mlfoundry_checkpoint_artifact_name automatically to {mlfoundry_checkpoint_artifact_name}"
                )
                other_arguments.mlfoundry_checkpoint_artifact_name = mlfoundry_checkpoint_artifact_name

        run.log_params(vars(other_arguments), flatten_params=True)
        run.log_params(
            filter_trainer_args_for_logging(training_arguments, other_arguments),
            flatten_params=True,
        )
        # TODO: there are 110 params in training_arguments, we do not need to log all of them.
        # run.log_params(training_arguments.to_sanitized_dict(), flatten_params=True)

    # Disk space management
    if dist_s.is_main_process:
        if other_arguments.cleanup_output_dir_on_start and os.path.exists(training_arguments.output_dir):
            logger.warning(f"--cleanup_output_dir_on_start was to set to True, wiping {training_arguments.output_dir}")
            shutil.rmtree(training_arguments.output_dir)

    if dist_s.is_main_process:
        if other_arguments.use_lora or other_arguments.use_qlora:
            with deepspeed_zero3_disabled(training_arguments):
                check_if_model_will_fit_only_with_gpus(
                    training_arguments=training_arguments, other_arguments=other_arguments
                )

    _train(
        training_arguments=training_arguments,
        other_arguments=other_arguments,
        run=run,
    )
    _cleanup_gpus()

    if dist_s.is_main_process:
        if other_arguments.use_lora or other_arguments.use_qlora:
            with deepspeed_zero3_disabled(training_arguments):
                merge_adapters_if_any(training_arguments=training_arguments, other_arguments=other_arguments)

    if dist_s.is_main_process and run:
        *_, model_name = other_arguments.model_id.rsplit("/", 1)
        model_name = "-".join(["finetuned", model_name, timestamp])
        model_name = sanitize_name(model_name)
        cleanup_checkpoints(output_dir=training_arguments.output_dir)
        log_model_to_mlfoundry(
            run=run,
            model_name=model_name,
            model_dir=training_arguments.output_dir,
            hf_hub_model_id=other_arguments.model_id,
            metadata=training_arguments.to_sanitized_dict() or {},
        )
        run.end()


def main():
    parser = HfArgumentParser(
        (HFTrainingArguments, OtherArguments),
        description="Fine-tune a language model on a text dataset",
    )
    training_arguments, other_arguments = parser.parse_args_into_dataclasses()
    train(
        training_arguments=training_arguments,
        other_arguments=other_arguments,
    )


if __name__ == "__main__":
    main()
