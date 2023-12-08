import copy
import gc
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import bitsandbytes as bnb
import mlfoundry
import numpy as np
import torch
import torch.backends.cuda
from accelerate import Accelerator, infer_auto_device_map, init_empty_weights
from accelerate.state import AcceleratorState
from cloudfiles import CloudFile
from datasets import Dataset, DatasetDict
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from huggingface_hub import scan_cache_dir
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from peft.tuners.lora import LoraLayer
from sklearn.model_selection import train_test_split
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    IntervalStrategy,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import rewrite_logs
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import WEIGHTS_NAME, is_torch_tf32_available
from transformers.utils import logging as hf_logging_utils

# TODO (chiragjn):
#   - Refactor and split code into sub modules.
#   - Try using deepspeed (with resume) for all 3 modes - qlora, lora and full
#   - Test and fix Deepspeed (Zero 3) weight gathering bugs during checkpointing if any
#   - Add support for dataset packing
#   - Find optimal combinations of batch_size, gradient accumulation, gradient checkpointing to get fastest training time in the given gpu budget
#   - Add support for dataset streaming
#   - Add support to use Apex FusedAdam
#   - Add support to push to HF Hub

MLFOUNDRY_ARTIFACT_PREFIX = "artifact:"
TFY_INTERNAL_JOB_NAME = os.getenv("TFY_INTERNAL_COMPONENT_NAME")
TFY_INTERNAL_JOB_RUN_NAME = os.getenv("TFY_INTERNAL_JOB_RUN_NAME")
THIS_DIR = os.path.abspath(os.path.dirname(__name__))
CACHE_DIR = os.path.join(THIS_DIR, ".cache")
EXPORT_ZERO3_CHECKPOINT_TO_FP32 = False
logger = logging.getLogger("truefoundry-finetune")

IGNORE_INDEX = -100  # -100 is the default ignore index in CrossEntropyLoss
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


class DataValidationException(Exception):
    pass


def get_torch_dtype(training_arguments: HFTrainingArguments):
    torch_dtype = None
    if training_arguments.bf16:
        torch_dtype = torch.bfloat16
    elif training_arguments.fp16:
        torch_dtype = torch.float16
    return torch_dtype


# --- Model checkpointing, saving and logging utils ---


def _is_mlfoundry_artifact(value: str):
    # TODO (chiragjn): This should be made more strict
    if value.startswith(MLFOUNDRY_ARTIFACT_PREFIX):
        return True


def _download_mlfoundry_artifact(artifact_version_fqn: str, download_dir: str, move_to: Optional[str] = None):
    client = mlfoundry.get_client()
    artifact_version = client.get_artifact_version_by_fqn(artifact_version_fqn)
    files_dir = artifact_version.download(download_dir)
    if move_to:
        files_dir = shutil.move(files_dir, move_to)
    return files_dir


def download_last_checkpoint_if_present(ml_repo: str, checkpoint_artifact_name: str, local_dir: str) -> Optional[str]:
    mlfoundry_client = mlfoundry.get_client()
    try:
        # TODO (chiragjn): We can use `:latest` tag
        latest_checkpoint_artifact = next(
            mlfoundry_client.list_artifact_versions(ml_repo=ml_repo, name=checkpoint_artifact_name)
        )
    except StopIteration:
        logger.info(
            f"No previous checkpoints found at artifact={checkpoint_artifact_name!r} in ml_repo={ml_repo!r}",
        )
        return
    # TODO: We should have specific exception to identify if the artifact does not exist
    except Exception as ex:
        logger.info("No previous checkpoints found. Message=%s", ex)
        return

    logger.info(
        "Downloading last checkpoint from artifact version=%r step=%r to resume training",
        latest_checkpoint_artifact.fqn,
        latest_checkpoint_artifact.step,
    )
    os.makedirs(local_dir, exist_ok=True)
    checkpoint_dir = os.path.join(local_dir, f"checkpoint-{latest_checkpoint_artifact.step}")
    with tempfile.TemporaryDirectory() as temp_dir:
        _download_mlfoundry_artifact(
            artifact_version_fqn=latest_checkpoint_artifact.fqn, download_dir=temp_dir, move_to=checkpoint_dir
        )
    return checkpoint_dir


def get_checkpoint_for_resume_if_any(
    training_arguments: HFTrainingArguments,
    other_arguments: OtherArguments,
) -> Optional[str]:
    accelerator_s = AcceleratorState()
    last_checkpoint_info_path = os.path.join(CACHE_DIR, "last_checkpoint_info.json")
    last_checkpoint_dir = None
    if accelerator_s.is_main_process:
        check_mlfoundry = False
        # resume_from_checkpoint can be None/true/false/string, None is default
        if training_arguments.resume_from_checkpoint is None:
            check_mlfoundry = True
        elif isinstance(training_arguments.resume_from_checkpoint, str):
            if os.path.exists(training_arguments.resume_from_checkpoint):
                last_checkpoint_dir = training_arguments.resume_from_checkpoint

            # TODO (chiragjn): Add support for resuming from an already saved checkpoint outside of the job run
            #   Although this is risky, because all other args (model, data, state) should remain same for a "correct" resume
            #   Note: Instead if we just want to resume from last checkpoint of the same job run then just use --mlfoundry_enable_reporting true --mlfoundry_checkpoint_artifact_name <name>
            # elif _is_mlfoundry_artifact(training_arguments.resume_from_checkpoint):
            #     _download_mlfoundry_artifact(...)

        elif training_arguments.resume_from_checkpoint is True:
            # Try locating latest checkpoint from output dir first
            if os.path.exists(training_arguments.output_dir):
                possible_last_checkpoint_dir = get_last_checkpoint(training_arguments.output_dir)
                if possible_last_checkpoint_dir:
                    last_checkpoint_dir = possible_last_checkpoint_dir

            if not last_checkpoint_dir:
                check_mlfoundry = True

        if check_mlfoundry and other_arguments.mlfoundry_enable_reporting:
            logger.info("Checking for any past checkpoints from same job run...")
            if other_arguments.mlfoundry_checkpoint_artifact_name:
                last_checkpoint_dir = download_last_checkpoint_if_present(
                    ml_repo=other_arguments.mlfoundry_ml_repo,
                    checkpoint_artifact_name=other_arguments.mlfoundry_checkpoint_artifact_name,
                    local_dir=training_arguments.output_dir,
                )

        with open(last_checkpoint_info_path, "w") as f:
            last_checkpoint_info = {"last_checkpoint_dir": last_checkpoint_dir}
            json.dump(last_checkpoint_info, f)
    else:
        with open(last_checkpoint_info_path, "r") as f:
            last_checkpoint_info = json.load(f)
        last_checkpoint_dir = last_checkpoint_info["last_checkpoint_dir"]
    return last_checkpoint_dir


def cleanup_checkpoints(
    training_arguments: HFTrainingArguments,
):
    logger.info("Cleaning up older checkpoints...")
    for f in os.listdir(training_arguments.output_dir):
        f_path = os.path.join(training_arguments.output_dir, f)
        if os.path.isdir(f_path) and f.startswith("checkpoint-"):
            shutil.rmtree(f_path)


def _cleanup_gpus():
    # TODO (chiragjn): We do not want anything to be offloaded to cpu or disk otherwise merging adapter fails!
    # This is a known issue with fix in progress
    #   - https://github.com/huggingface/peft/pull/1063
    #   - https://github.com/huggingface/transformers/pull/27412
    # Yes, sleeping is stupid but necessary till the above PRs are merged and made available in a new version
    for _ in range(5):
        gc.collect()
        time.sleep(3)
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
        device_map="sequential",
    )
    logger.info("Merging lora adapter into main model. This can take a while ...")
    model = model.merge_and_unload()
    model.save_pretrained(training_arguments.output_dir, safe_serialization=True)
    for filename in ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]:
        file_to_delete = os.path.join(training_arguments.output_dir, filename)
        if os.path.exists(file_to_delete):
            os.remove(file_to_delete)


def log_model_to_mlfoundry(
    run: mlfoundry.MlFoundryRun, training_arguments: HFTrainingArguments, model_name: str, hf_hub_model_id: str
):
    logger.info("Uploading Model...")
    cleanup_checkpoints(training_arguments=training_arguments)

    hf_cache_info = scan_cache_dir()
    files_to_save = []
    for repo in hf_cache_info.repos:
        if repo.repo_id == hf_hub_model_id:
            for revision in repo.revisions:
                for file in revision.files:
                    if file.file_path.name.endswith(".py"):
                        files_to_save.append(file.file_path)
                break

    # copy the files to output_dir of pipeline
    for file_path in files_to_save:
        match = re.match(r".*snapshots\/[^\/]+\/(.*)", str(file_path))
        if match:
            relative_path = match.group(1)
            destination_path = os.path.join(training_arguments.output_dir, relative_path)
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.copy(str(file_path), destination_path)
        else:
            logger.warning("Python file in hf model cache in unknown path:", file_path)

    metadata = training_arguments.to_sanitized_dict()
    metadata.update({"huggingface_model_url": f"https://huggingface.co/{hf_hub_model_id}"})

    run.log_model(
        name=model_name,
        model_file_or_folder=training_arguments.output_dir,
        framework="transformers",
        metadata=metadata,
    )


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
    if other_arguments.use_lora:
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
            "bnb_4bit_quant_type": other_arguments.bnb_4bit_quant_type,
            "use_double_quant": other_arguments.use_double_quant,
            "qlora_bit_length": other_arguments.qlora_bit_length,
        }
        arguments.update(qlora_args)

    return arguments


class MLFoundryCallback(TrainerCallback):
    def __init__(
        self,
        run: Optional[mlfoundry.MlFoundryRun] = None,
        checkpoint_artifact_name: Optional[str] = None,
        log_checkpoints: bool = True,
    ):
        self._run = run
        self._checkpoint_artifact_name = checkpoint_artifact_name
        self._log_checkpoints = log_checkpoints

        if not self._checkpoint_artifact_name:
            logger.warning("checkpoint_artifact_name not passed. Checkpoints will not be logged to MLFoundry")

    # noinspection PyMethodOverriding
    def on_log(self, args, state, control, logs, model=None, **kwargs):
        # TODO (chiragjn): Hack for now, needs to be moved to `compute_metrics`
        #   unfortunately compute metrics does not give us already computed metrics like eval_loss
        if not state.is_world_process_zero:
            return

        for loss_key, perplexity_key in [("loss", "train_perplexity"), ("eval_loss", "eval_perplexity")]:
            if loss_key in logs:
                try:
                    perplexity = math.exp(logs[loss_key])
                except OverflowError:
                    perplexity = float("inf")
                    logger.warning(f"Encountered inf in eval perplexity, cannot log it as a metric")
                logger.info(f"{perplexity_key}: {perplexity}")
                logs[perplexity_key] = perplexity

        logger.info(f"Metrics: {logs}")
        if not self._run:
            return

        metrics = {}
        for k, v in logs.items():
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

        if not self._run or not self._checkpoint_artifact_name:
            return

        ckpt_dir = f"checkpoint-{state.global_step}"
        artifact_path = os.path.join(args.output_dir, ckpt_dir)
        description = None
        if TFY_INTERNAL_JOB_NAME:
            description = f"Checkpoint from finetuning job={TFY_INTERNAL_JOB_NAME} run={TFY_INTERNAL_JOB_RUN_NAME}"
        logger.info(f"Uploading checkpoint {ckpt_dir} ...")
        self._run.log_artifact(
            name=self._checkpoint_artifact_name,
            artifact_paths=[(artifact_path,)],
            step=state.global_step,
            description=description,
        )


# --- Data Processing Utils ---


class DatasetBuilder:
    """Dataset agnostic class to take in input_ids and labels and spit out tokens"""

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def batch_tokenize(self, texts):
        """Tokenizes text. Presently doesn't pad inputs, just returns input ids."""
        tokenized = [
            self.tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=self.max_length,
                truncation=True,
            ).input_ids
            for text in texts
        ]
        return tokenized

    def construct_dataset(self, input_batch):
        tokenized_input_ids = self.batch_tokenize(input_batch[PROMPT_KEY])
        tokenized_labels = self.batch_tokenize(input_batch[COMPLETION_KEY])
        return {"input_ids": tokenized_input_ids, "labels": tokenized_labels}


class CausalDatasetBuilder(DatasetBuilder):
    """Builds generative dataset for Causal LM."""

    def __init__(self, tokenizer, max_length, train_on_prompt=True):
        super().__init__(tokenizer, max_length)
        self.train_on_prompt = train_on_prompt

    def construct_dataset(self, input_batch):
        labels = []
        for prompt, completion in zip(input_batch[PROMPT_KEY], input_batch[COMPLETION_KEY]):
            labels.append(prompt + "\n" + completion + self.tokenizer.eos_token)
        input_ids = [val.squeeze() for val in self.batch_tokenize(labels)]
        labels = copy.deepcopy(input_ids)
        if not self.train_on_prompt:
            tokenized_prompts = self.batch_tokenize(input_batch[PROMPT_KEY])
            prompt_lens = [val.shape[1] for val in tokenized_prompts]
            for label, source_len in zip(labels, prompt_lens):
                label[:source_len] = IGNORE_INDEX
        return {"input_ids": input_ids, "labels": labels}


class SequenceDataCollator:
    """Collate examples for dynamic batch construction in supervised fine-tuning."""

    def __init__(self, tokenizer, multiple_of=None):
        self.tokenizer = tokenizer
        self.multiple_of = multiple_of

    def pad_to_multiple(self, tensor, value):
        multiple = self.multiple_of
        n = tensor.size(-1)
        target_length = (n + multiple - 1) // multiple * multiple
        return torch.nn.functional.pad(tensor, (0, target_length - n), value=value)

    def __call__(self, instances):
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )  # -100 tells torch to ignore these tokens in loss computation.
        if self.multiple_of:
            input_ids = self.pad_to_multiple(input_ids, value=self.tokenizer.pad_token_id)
            labels = self.pad_to_multiple(labels, value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def _read_lines_from_files(download_path):
    for root, dirs, files in os.walk(download_path):
        for file in files:
            filepath = os.path.join(root, file)
            filename = os.path.basename(filepath)
            if filename.endswith(".jsonl") and not filename.startswith("."):
                logger.info(f"Loading file {filename} ...")
                with open(filepath) as f:
                    for line in f.readlines():
                        yield line


def _read_lines_from_cloudfile(path):
    raw_data = CloudFile(path).get().decode("utf-8").split("\n")
    for line in raw_data:
        yield line


def load_data(path, max_num_samples: Optional[int] = None):
    data = []
    n = max_num_samples if max_num_samples else -1
    count = 0
    with tempfile.TemporaryDirectory() as download_dir:
        if _is_mlfoundry_artifact(path):
            logger.info("Downloading artifact from mlfoundry")
            download_path = _download_mlfoundry_artifact(artifact_version_fqn=path, download_dir=download_dir)
            lines = _read_lines_from_files(download_path)
        elif path.startswith("snowflake://"):
            from utils import get_data_from_snowflake_table

            logger.info(f"Loading data from snowflake db ...")
            lines = get_data_from_snowflake_table(uri=path, max_num_samples=max_num_samples)
        else:
            logger.info(f"Loading data from link: {path}")
            lines = _read_lines_from_cloudfile(path)
        for line_no, line in enumerate(lines, start=1):
            if n > 0 and count >= n:
                break
            if not line.strip():
                continue
            try:
                datapoint_dict = json.loads(line)
            except json.decoder.JSONDecodeError as je:
                raise DataValidationException(
                    f"Failed to parse json line on line number {line_no}. Line: {line[:150]}..."
                ) from je
            else:
                for key in (PROMPT_KEY, COMPLETION_KEY):
                    if key not in datapoint_dict:
                        raise DataValidationException(
                            f"Required key `{key}` is missing from json line on line number {line_no}. Line: {line[:150]}..."
                        )
                    if not isinstance(datapoint_dict[key], str):
                        raise DataValidationException(
                            f"Value for `{key}` is not string on line number {line_no}. Line: {line[:150]}..."
                        )

                datapoint_dict = {
                    PROMPT_KEY: datapoint_dict[PROMPT_KEY],
                    COMPLETION_KEY: datapoint_dict[COMPLETION_KEY],
                }
                data.append(datapoint_dict)
                count += 1
    return data


def get_data(training_arguments: HFTrainingArguments, other_arguments: OtherArguments):
    logger.info(f"Loading train dataset ...")
    train_data = load_data(other_arguments.train_data, max_num_samples=other_arguments.max_num_samples)
    eval_data = other_arguments.eval_data
    if eval_data and eval_data != "NA":
        logger.info(f"Loading eval dataset {other_arguments.eval_data}...")
        eval_data = load_data(eval_data, max_num_samples=other_arguments.max_num_samples)
    elif other_arguments.eval_size:
        logger.info(f"No eval dataset given, splitting from training dataset...")
        train_data, eval_data = train_test_split(
            train_data,
            test_size=other_arguments.eval_size,
            random_state=training_arguments.data_seed,
        )
    return train_data, eval_data


def build_dataset(
    train_data,
    eval_data,
    tokenizer,
    max_length: int,
    training_arguments: TrainingArguments,
    other_arguments: OtherArguments,
):
    accelerator_s = AcceleratorState()
    logger.info("Building dataset...")
    dataset_cache_path = os.path.join(CACHE_DIR, "dataset")
    if accelerator_s.is_main_process:
        builder = CausalDatasetBuilder(
            tokenizer=tokenizer, max_length=max_length, train_on_prompt=other_arguments.train_on_prompt
        )
        dataset_dict = DatasetDict(train=Dataset.from_list(train_data), eval=Dataset.from_list(eval_data))
        # TODO (chiragjn): Read cpu limits from cgroup, cpu_count is not usable in containers environment
        num_proc = max(1, min(4, os.cpu_count()))
        num_proc = num_proc if num_proc > 1 else None
        dataset_dict = dataset_dict.map(
            builder.construct_dataset,
            remove_columns=[PROMPT_KEY, COMPLETION_KEY],
            batched=True,
            batch_size=32,
            num_proc=num_proc,
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


# --- Core Training Code ---


def _log_model_parameters(model):
    logger.info("=== Model Parameters ===")
    for name, parameter in model.named_parameters():
        print("\t", name, parameter.dtype, parameter.device, parameter.requires_grad)
    logger.info("========================")


def _maybe_set_custom_tempdir():
    # We make sure any custom tempdir set by setting `TMPDIR` or equivalent env variables exist
    _tempdir = os.getenv("TMPDIR")
    if _tempdir:
        if os.path.exists(_tempdir) and os.path.isfile(_tempdir):
            raise ValueError("Current `TMPDIR` points to a file path, please set it to a directory path")
        else:
            os.makedirs(_tempdir, exist_ok=True)


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
        torch.cuda.set_per_process_memory_fraction(0.9, device=device)


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
        # if not (training_arguments.bf16 or training_arguments.fp16):
        #     raise ValueError("--use_flash_attention requires either --bf16 or --fp16")
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
    accelerator_s = AcceleratorState()
    logger.info("Loading model...")
    model_load_kwargs = {}
    model_load_kwargs["use_cache"] = False if training_arguments.gradient_checkpointing else True
    if model_config.architectures and model_config.architectures[0] == "PhiForCausalLM":
        model_load_kwargs.pop("use_cache", None)

    if other_arguments.use_flash_attention:
        model_load_kwargs["use_flash_attention_2"] = other_arguments.use_flash_attention

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
        if accelerator_s.is_main_process:
            _log_model_parameters(model)
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_arguments.gradient_checkpointing
        )
        if accelerator_s.is_main_process:
            _log_model_parameters(model)
        # TODO (chiragjn): This is disabled because resuming does not work: https://github.com/TimDettmers/bitsandbytes/issues/782
        # training_arguments.optim = "paged_adamw_32bit"
    else:
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
    acclerator_s = AcceleratorState()
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
        if any(ename in name for ename in ("lm_head", "embed_tokens", "embed_in", "embed_out", "wte", "wpe")):
            if hasattr(module, "weight"):
                if training_arguments.bf16 and module.weight.dtype == torch.float32:
                    # This is experimental, normally qlora repo uses it but some others don't recommend it. So far we haven't see major problems.
                    # Note this downcasting is not recommended when using float16, that can cause numerical instability
                    module = module.to(torch.bfloat16)

    model.enable_input_require_grads()
    model.print_trainable_parameters()
    if acclerator_s.is_main_process:
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


def check_if_model_will_fit_only_with_gpus(
    training_arguments: HFTrainingArguments,
    other_arguments: OtherArguments,
):
    config = AutoConfig.from_pretrained(other_arguments.model_id, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    device_map = infer_auto_device_map(model, dtype=get_torch_dtype(training_arguments))
    logger.info(f"Inferred device_map for auto settings: {device_map}")
    if any(not isinstance(v, int) for v in device_map.values()):
        raise RuntimeError(
            "For lora/qlora the model must entirely fit on gpus without any kind of offloading to prevent bugs with merging! "
            "With the current configuration model is being offloaded to cpu/disk. This causes incorrect model saving. See https://github.com/huggingface/peft/issues/868"
        )


def _train(
    *,
    training_arguments: HFTrainingArguments,
    other_arguments: OtherArguments,
    run: Optional[mlfoundry.MlFoundryRun] = None,
):
    accelerator_s = AcceleratorState()
    set_seed(training_arguments.seed)

    if not accelerator_s.is_main_process:
        logger.info("Waiting for main process to load data, process it and fetch any checkpoints ...")

    with accelerator_s.main_process_first():
        if accelerator_s.is_main_process:
            train_data, eval_data = get_data(training_arguments=training_arguments, other_arguments=other_arguments)
        else:
            train_data, eval_data = None, None

        last_checkpoint_dir = get_checkpoint_for_resume_if_any(
            training_arguments=training_arguments,
            other_arguments=other_arguments,
        )

        logger.info("Loading config ...")
        model_config = AutoConfig.from_pretrained(other_arguments.model_id, trust_remote_code=True)

        if last_checkpoint_dir:
            model_source = last_checkpoint_dir
        else:
            model_source = other_arguments.model_id

        tokenizer, num_new_tokens = get_tokenizer(model_source)

        max_length = get_max_length(
            max_length=other_arguments.max_length, tokenizer=tokenizer, model_config=model_config
        )

        train_dataset, eval_dataset = build_dataset(
            train_data=train_data,
            eval_data=eval_data,
            tokenizer=tokenizer,
            max_length=max_length,
            training_arguments=training_arguments,
            other_arguments=other_arguments,
        )

        if accelerator_s.is_main_process:
            logger.info("Getting other ranks in sync with main process")

    no_of_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_map = None
    if not training_arguments.deepspeed:
        if other_arguments.use_ddp and no_of_gpus > 1:
            device_map = {"": "cuda:" + str(training_arguments.local_rank)}
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
    #   There is a massive refactor in peft which has mostly solved this but unreleased as of writing: https://github.com/huggingface/peft/commit/5a3a5acff2d679358251742564f7b12efbee3a41
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
        logger.info("Resizing embeddings layer for newly added tokens")
        model.resize_token_embeddings(len(tokenizer))

    if other_arguments.use_lora or other_arguments.use_qlora:
        model = get_peft_wrapped_model(
            model,
            training_arguments=training_arguments,
            other_arguments=other_arguments,
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    logger.info("Training...")
    # TODO (chiragjn): Add text generation metrics to `compute_metrics
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_arguments,
        data_collator=SequenceDataCollator(tokenizer, multiple_of=other_arguments.pad_to_multiple_of),
        callbacks=[
            MLFoundryCallback(
                run=run,
                checkpoint_artifact_name=other_arguments.mlfoundry_checkpoint_artifact_name,
                log_checkpoints=other_arguments.mlfoundry_log_checkpoints,
            )
        ],
    )

    trainer.train(resume_from_checkpoint=last_checkpoint_dir)

    accelerator_s.wait_for_everyone()

    logger.info("Saving model...")
    if training_arguments.deepspeed and is_deepspeed_zero3_enabled() and EXPORT_ZERO3_CHECKPOINT_TO_FP32:
        # TODO (chiragjn): Disabled for now. Test and Re-enable, check the half precision format
        #  Under ZeRO 3, when checkpointing, each rank saves their own part, in zero format
        #  if "stage3_gather_16bit_weights_on_model_save": true,
        #  then an additional pytorch_model.bin is saved as a 16-bit checkpoint
        #  if we want fp32 pytorch_model.bin then we would have to export separately from the checkpoint in zero format
        trainer.save_model(output_dir=training_arguments.output_dir)
        if accelerator_s.is_main_process:
            fp32_weights_path = os.path.join(training_arguments.output_dir, WEIGHTS_NAME)
            convert_zero_checkpoint_to_fp32_state_dict(trainer.state.best_model_checkpoint, fp32_weights_path)
            cleanup_checkpoints(training_arguments=training_arguments)
    else:
        if accelerator_s.is_main_process:
            cleanup_checkpoints(training_arguments=training_arguments)
        trainer.save_model(output_dir=training_arguments.output_dir)

    accelerator_s.wait_for_everyone()


def train(training_arguments: HFTrainingArguments, other_arguments: OtherArguments):
    accelerator = Accelerator()
    accelerator_s = AcceleratorState()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    logger.info(f"Training Arguments: {training_arguments}")
    logger.info(f"Arguments: {other_arguments}")

    if other_arguments.use_lora or other_arguments.use_qlora:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("No GPUs detected. We need at least one gpu available for Lora/QLora finetuning!")
        # TODO (chiragjn): Support LoRA and QLoRA with deepspeed
        if training_arguments.deepspeed:
            raise ValueError(
                "deepspeed is currently not supported with lora/qlora fine-tuning please try fine-tuning without deepspeed"
            )

    setup(training_arguments=training_arguments, other_arguments=other_arguments)

    run = None
    if accelerator_s.is_main_process and other_arguments.mlfoundry_enable_reporting:
        mlfoundry_client = mlfoundry.get_client()
        if not other_arguments.mlfoundry_run_name:
            fallback_run_name = f"finetune-{timestamp}"
            logger.info(f"Setting --mlfoundry_run_name automatically to {fallback_run_name}")
            other_arguments.mlfoundry_run_name = fallback_run_name
        run = mlfoundry_client.create_run(
            ml_repo=other_arguments.mlfoundry_ml_repo, run_name=other_arguments.mlfoundry_run_name
        )

        if not other_arguments.mlfoundry_checkpoint_artifact_name:
            if TFY_INTERNAL_JOB_RUN_NAME:
                mlfoundry_checkpoint_artifact_name = f"checkpoint-{TFY_INTERNAL_JOB_RUN_NAME}"
                logger.info(
                    f"Setting --mlfoundry_checkpoint_artifact_name automatically to {mlfoundry_checkpoint_artifact_name}"
                )
                other_arguments.mlfoundry_checkpoint_artifact_name = mlfoundry_checkpoint_artifact_name

        if other_arguments.mlfoundry_log_checkpoints and not other_arguments.mlfoundry_checkpoint_artifact_name:
            raise ValueError(
                "--mlfoundry_log_checkpoints was set to true but --mlfoundry_checkpoint_artifact_name is either unset or cannot be automatically decided. Please set it explicitly"
            )

        run.log_params(vars(other_arguments), flatten_params=True)
        run.log_params(filter_trainer_args_for_logging(training_arguments, other_arguments), flatten_params=True)
        # TODO: there are 110 params in training_arguments, we do not need to log all of them.
        # run.log_params(training_arguments.to_sanitized_dict(), flatten_params=True)

    # Disk space management
    if accelerator_s.is_main_process:
        if other_arguments.cleanup_output_dir_on_start and os.path.exists(training_arguments.output_dir):
            logger.warning(f"--cleanup_output_dir_on_start was to set to True, wiping {training_arguments.output_dir}")
            shutil.rmtree(training_arguments.output_dir)

    if accelerator_s.is_main_process:
        if other_arguments.use_lora or other_arguments.use_qlora:
            check_if_model_will_fit_only_with_gpus(
                training_arguments=training_arguments, other_arguments=other_arguments
            )

    _train(
        training_arguments=training_arguments,
        other_arguments=other_arguments,
        run=run,
    )
    _cleanup_gpus()

    if accelerator_s.is_main_process:
        if other_arguments.use_lora or other_arguments.use_qlora:
            merge_adapters_if_any(training_arguments=training_arguments, other_arguments=other_arguments)

    if accelerator_s.is_main_process and run:
        *_, model_name = other_arguments.model_id.rsplit("/", 1)
        model_name = "-".join(["finetuned", model_name, timestamp])
        model_name = model_name.replace(".", "-")
        log_model_to_mlfoundry(
            run=run,
            training_arguments=training_arguments,
            model_name=model_name,
            hf_hub_model_id=other_arguments.model_id,
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
