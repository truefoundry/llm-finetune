#!/bin/bash

# --- Environment variables ---
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,roundup_power2_divisions:16"

## This controls how many GPUs you want to use
export CUDA_VISIBLE_DEVICES=0
## This controls how much memory to user per gpu
export TORCH_PER_PROCESS_MEMORY_LIMIT=0.98

## Add your token for private/gated models
export HF_TOKEN=

## Turn these on for debugging
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=INFO

# --- Agruments ----

## If to delete outputs/ dir before starting - to start from clean slate
CLEANUP_OUTPUT_DIR_ON_START=True

## You can logs metrics, checkpoints and final model with TrueFoundry Experiment Tracking
TRUEFOUNDRY_ML_ENABLE_REPORTING=False
TRUEFOUNDRY_ML_REPO=llm-finetuning
TRUEFOUNDRY_ML_RUN_NAME=my-finetuning-run-name-1

accelerate launch \
--mixed_precision bf16 \
--use_deepspeed \
train.py \
config-base.yaml \
--deepspeed ./deepspeed_configs/3_ds_z2_config.json \
--base_model meta-llama/Llama-3.2-1B-Instruct \
--train_data_uri ./sample_data/chatalpaca-openai-100.jsonl \
--val_data_uri None \
--val_set_size 0.2 \
--dataset_type chat \
--sequence_len 4096 \
--max_steps 0 \
--micro_batch_size 2 \
--eval_batch_size 2 \
--num_epochs 3 \
--gradient_accumulation_steps 4 \
--gradient_checkpointing unsloth \
--learning_rate 0.00001 \
--output_dir ./outputs \
--train_on_inputs False \
--logging_steps 1 \
--save_strategy steps \
--save_steps 0.2 \
--evaluation_strategy steps \
--eval_steps 0.2 \
--adapter qlora \
--lora_target_linear True \
--lora_r 128 \
--lora_alpha 256 \
--truefoundry_ml_enable_reporting $TRUEFOUNDRY_ML_ENABLE_REPORTING \
--truefoundry_ml_repo $TRUEFOUNDRY_ML_REPO \
--truefoundry_ml_run_name $TRUEFOUNDRY_ML_RUN_NAME \
--truefoundry_ml_log_checkpoints True \
--resume_from_checkpoint False \
--cleanup_output_dir_on_start $CLEANUP_OUTPUT_DIR_ON_START
