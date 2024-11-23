"""
A very hacky script to test out capturing GPU memory usage against tokens and trainable parameters
Later this will be more automated and parallelized across TrueFoundry Jobs
"""
import argparse
import itertools
import json
import os
import shlex
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List

import pandas as pd
import yaml
from pydantic import BaseModel
from transformers import AutoConfig


class ReportingConfig(BaseModel):
    ml_repo: str = "llm-ft-reporting"
    base_models: List[str] = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
    ]
    sequence_lens: List[int] = [512, 1024, 2048, 4096, 8192]
    lora_rs: List[int] = [32]
    stream_stdout: bool = False
    stream_stderr: bool = False


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
--val_set_size 0.2
--sequence_len {sequence_len}
--long_sequences_strategy drop
--micro_batch_size 1
--eval_batch_size 1
--eval_sample_packing True
--num_epochs 1
--max_steps 3
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
--merge_adapters_post_train False
"""


def stream_output(pipe, prefix=""):
    for line in iter(pipe.readline, ""):
        print(f"{prefix}{line.strip()}")
    pipe.close()


def run_command(command: List[str], stream_stdout=False, stream_stderr=False):
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
            futures = []
            if stream_stdout:
                futures.append(executor.submit(stream_output, process.stdout, "STDOUT: "))
            if stream_stderr:
                futures.append(executor.submit(stream_output, process.stderr, "STDERR: "))
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="reporting_config.yaml")
    parser.add_argument("--output", type=str, default="report.csv")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = ReportingConfig.model_validate(config)
    env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,roundup_power2_divisions:16",
        "CUDA_VISIBLE_DEVICES": "0",
        "TORCH_PER_PROCESS_MEMORY_LIMIT": "0.98",
        "GPU_CLEANUP_N_ITERS": "3",
        "GPU_CLEANUP_INTERVAL_SECONDS": "3",
    }
    for k, v in env.items():
        os.environ[k] = v

    reports = []
    for model, seq_len, lora_r in itertools.product(config.base_models, config.sequence_lens, config.lora_rs):
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
            ml_repo=config.ml_repo,
            run_name=run_name,
        )
        try:
            run_command(
                shlex.split(command),
                stream_stdout=config.stream_stdout,
                stream_stderr=config.stream_stderr,
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
        model_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        report = {
            "base_model": model,
            "seq_len": seq_len,
            "lora_r": lora_r,
            "trainable_params": trainable_params,
            "all_params": all_params,
            "cuda_oom": cuda_oom,
            "max_gpu_memory_allocated": max_gpu_memory_allocated,
            "model_config": json.loads(model_config.to_json_string()),
        }
        print("=" * 80)
        print(json.dumps(report))
        print("=" * 80)
        reports.append(report)
        if not trainable_params or not all_params:
            raise Exception("Failed to capture params")

        if not cuda_oom and max_gpu_memory_allocated == -1:
            raise Exception("Failed to capture GPU memory usage")

    df = pd.DataFrame(reports)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
