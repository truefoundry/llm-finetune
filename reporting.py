"""
A very hacky script to test out capturing GPU memory usage against tokens and trainable parameters
Later this will be more automated and parallelized across TrueFoundry Jobs
"""
import itertools
import json
import os
import shlex
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List

from transformers import AutoConfig

ML_REPO = "llm-ft-reporting"

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
]
SEQ_LENS = [512, 1024, 2048, 4096, 8192]
LORA_RS = [8, 16, 32]

COMMAND = """\
accelerate launch
--mixed_precision bf16
--use_deepspeed
train.py
config-base.yaml
--deepspeed ./deepspeed_configs/3_ds_z2_config.json
--base_model {base_model}
--dataset_type chat
--train_data_uri ./sample_data/chatalpaca-openai-1k.jsonl
--val_data_uri None
--val_set_size 0.1
--sequence_len {sequence_len}
--long_sequences_strategy drop
--micro_batch_size 1
--eval_batch_size 1
--num_epochs 1
--max_steps 10
--gradient_accumulation_steps 4
--gradient_checkpointing unsloth
--learning_rate 0.00001
--output_dir ./outputs
--train_on_inputs False
--logging_steps 1
--save_strategy steps
--save_steps 0.5
--evaluation_strategy steps
--eval_steps 0.5
--adapter qlora
--lora_target_linear True
--lora_r {lora_r}
--lora_alpha {lora_alpha}
--resume_from_checkpoint False
--cleanup_output_dir_on_start True
--pad_to_sequence_len True
--truefoundry_ml_enable_reporting True
--truefoundry_ml_repo {ml_repo}
--truefoundry_ml_run_name {run_name}
--truefoundry_ml_log_checkpoints False
--truefoundry_ml_log_gpu_metrics True
--truefoundry_ml_log_merged_model False
--truefoundry_testing_mode True
"""


def stream_output(pipe, prefix=""):
    for line in iter(pipe.readline, ""):
        print(f"{prefix}{line.strip()}")
    pipe.close()


def run_command(command: List[str]):
    print("Running command: ", " ".join(command))
    try:
        process = subprocess.Popen(
            shlex.join(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
            env=os.environ,
            shell=True,
        )
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(stream_output, process.stdout, "STDOUT: "),
                # executor.submit(stream_output, process.stderr, "STDERR: "),
            ]
            process.wait()
            for future in futures:
                future.result()
            if process.returncode != 0:
                raise Exception(f"Command failed with return code {process.returncode}")
    except subprocess.CalledProcessError as e:
        raise Exception(f"An error occurred while executing the command: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")


def main():
    env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,roundup_power2_divisions:16",
        "CUDA_VISIBLE_DEVICES": "0",
        "TORCH_PER_PROCESS_MEMORY_LIMIT": "0.98",
    }
    for k, v in env.items():
        os.environ[k] = v
    for model, seq_len, lora_r in itertools.product(MODELS, SEQ_LENS, LORA_RS):
        if os.path.exists("axolotl_truefoundry.plugin.log"):
            os.remove("axolotl_truefoundry.plugin.log")
        if os.path.exists("train.log"):
            os.remove("train.log")
        print(f"Model: {model}, Seq Len: {seq_len}, LoRA R: {lora_r}")
        run_name = str(uuid.uuid4())
        command = COMMAND.format(
            base_model=model,
            sequence_len=str(seq_len),
            lora_r=str(lora_r),
            lora_alpha=str(lora_r * 2),
            ml_repo=ML_REPO,
            run_name=run_name,
        )
        try:
            run_command(
                shlex.split(command),
            )
        except Exception as e:
            print(f"Failed to run command: {e}")

        logs = []
        with open("axolotl_truefoundry.plugin.log") as f:
            logs = [json.loads(line) for line in f.readlines()]
        trainable_params = None
        all_params = None
        max_gpu_memory_allocated = -1
        for log in logs:
            if "trainable_params" in log:
                trainable_params = log["trainable_params"]
            if "all_params" in log:
                all_params = log["all_params"]
            if "system/gpu.0.memory_allocated" in log:
                max_gpu_memory_allocated = max(max_gpu_memory_allocated, log["system/gpu.0.memory_allocated"])

        cuda_oom = False
        with open("train.log") as f:
            for line in f.readlines():
                if "CUDA out of memory. Tried to allocate" in line:
                    cuda_oom = True
                    break
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        print("=" * 80)
        print(f"Config: {config}")
        print(f"Model: {model}")
        print(f"Seq Len: {seq_len}")
        print(f"LoRA R: {lora_r}")
        print(f"Trainable Params: {trainable_params}")
        print(f"All Params: {all_params}")
        print(f"CUDA OOM: {cuda_oom}")
        print(f"GPU Memory Allocated: {max_gpu_memory_allocated}")
        print("=" * 80)
        if not trainable_params or not all_params:
            raise Exception("Failed to capture params")

        if not cuda_oom and max_gpu_memory_allocated == -1:
            raise Exception("Failed to capture GPU memory usage")


if __name__ == "__main__":
    main()
