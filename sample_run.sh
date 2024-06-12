#!/bin/bash

# --- Environment variables ---
export DISABLE_MLFLOW_INTEGRATION=True
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

## This controls how many GPUs you want to use
export CUDA_VISIBLE_DEVICES=0
## This controls how much memory to user per gpu
export TORCH_PER_PROCESS_MEMORY_LIMIT=0.99

## Add your token for private/gated models
# export HF_TOKEN=

## Turn these on for debugging
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=INFO

# --- Agruments ----

## If to delete outputs/ dir before starting - to start from clean slate
CLEANUP_OUTPUT_DIR_ON_START=True

## You can logs metrics, checkpoints and final model with TrueFoundry Experiment Tracking
MLFOUNDRY_ENABLE_REPORTING=False
MLFOUNDRY_ML_REPO=llm-finetuning
MLFOUNDRY_RUN_NAME=my-finetuning-run-name

accelerate launch \
--mixed_precision bf16 \
--use_deepspeed \
train.py \
config-base.yaml \
--deepspeed ./deepspeed_configs/3_ds_z2_config.json \
--base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
--train_data_uri ./sample_data/chatalpaca-openai-100.jsonl \
--val_data_uri None \
--val_set_size 0.1 \
--dataset_type chat \
--sequence_len 4096 \
--max_steps 0 \
--micro_batch_size 1 \
--eval_batch_size 1 \
--num_epochs 1 \
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
--lora_r 16 \
--lora_alpha 32 \
--mlfoundry_enable_reporting $MLFOUNDRY_ENABLE_REPORTING \
--mlfoundry_ml_repo $MLFOUNDRY_ML_REPO \
--mlfoundry_run_name $MLFOUNDRY_RUN_NAME \
--mlfoundry_log_checkpoints True \
--resume_from_checkpoint True \
--cleanup_output_dir_on_start $CLEANUP_OUTPUT_DIR_ON_START
