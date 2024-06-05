> [!important]
> Please prefer using commits from [release tags](https://github.com/truefoundry/llm-finetune/releases). `main` branch is work in progress and may have partially working commits.

## LLM Finetuning with Truefoundry
Test QLoRA w/ Deepspeed Stage 2

```
#!/bin/bash

# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=INFO
# export TORCH_PER_PROCESS_MEMORY_LIMIT=22000
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0
export DISABLE_MLFLOW_INTEGRATION=True

TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
LORA_R=32
LORA_ALPHA=64
TORCH_PER_PROCESS_MEMORY_LIMIT=0.95
CUDA_VISIBLE_DEVICES=0,1
TRAIN_DATA="./data/standford_alpaca_train_49k.jsonl"
# TRAIN_DATA="./data/lima_llama2_1k.jsonl"
MAX_STEPS=10
MODEL_ID=NousResearch/Llama-2-7b-chat-hf
USE_FLASH_ATTENTION=True
GRADIENT_CHECKPOINTING=True
NUM_TRAIN_EPOCHS=3


# --deepspeed ./deepspeed_configs/3_ds_z2_config.json \
# --deepspeed ./deepspeed_configs/4_ds_z2_offload_optimizer_config.json \
# --deepspeed ./deepspeed_configs/5_ds_z3_config.json \
# --deepspeed ./deepspeed_configs/6_ds_z3_offload_param_config.json \
# --deepspeed ./deepspeed_configs/7_ds_z3_offload_optimizer_config.json \
# --deepspeed ./deepspeed_configs/8_ds_z3_offload_param_offload_optimizer_config.json \

accelerate launch \
--mixed_precision bf16 \
--use_deepspeed \
train.py \
config-base.yaml \
--deepspeed ./deepspeed_configs/3_ds_z2_config.json \
--flash_attention $USE_FLASH_ATTENTION \
--base_model $MODEL_ID \
--train_data_uri $TRAIN_DATA \
--max_steps $MAX_STEPS \
--val_data_uri None \
--val_set_size 0.1 \
--micro_batch_size $TRAIN_BATCH_SIZE \
--num_epochs $NUM_TRAIN_EPOCHS \
--gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
--gradient_checkpointing $GRADIENT_CHECKPOINTING \
--learning_rate 0.00001 \
--output_dir ./outputs \
--train_on_inputs False \
--logging_steps 1 \
--save_strategy steps \
--save_steps 0.05 \
--evaluation_strategy steps \
--eval_steps 0.05 \
--adapter qlora \
--lora_target_linear True \
--lora_r $LORA_R \
--lora_alpha $LORA_ALPHA \
--mlfoundry_enable_reporting False \
--mlfoundry_ml_repo my-ml-repo \
--mlfoundry_run_name test \
--mlfoundry_checkpoint_artifact_name chk-test \
--mlfoundry_log_checkpoints False \
--resume_from_checkpoint False \
--cleanup_output_dir_on_start True
```


- `TORCH_PER_PROCESS_MEMORY_LIMIT` allows limiting the max memory per gpu. Can be a fraction (denoting percentage) or integer (denoting limit in MiB). Useful for testing limited gpu memory scenarios
- CUDA_VISIBLE_DEVICES can be used to control the amount of GPUs
- `--mlfoundry_enable_reporting true/false` toggles reporting metrics, checkpoints and models to mlfoundry
- When you are testing locally, you can set `--cleanup_output_dir_on_start true` if you don't care about checkpoints between runs

---

Generally we always try to optimize for memory footprint because that allows higher batch size and more gpu utilization
Speedup is second priority but we take what we can easily get
