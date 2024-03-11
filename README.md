Axolotl config options

<details>
    <summary>Click to expand all axolotl options</summary>

Just dumping here, because some options are not documented

```
cfg.adam_beta1
cfg.adam_beta2
cfg.adam_epsilon
cfg.adapter
cfg.auto_resume_from_checkpoints
cfg.axolotl_config_path
cfg.base_model
cfg.base_model_config
cfg.batch_size
cfg.bench_dataset
cfg.bf16
cfg.bfloat16
cfg.bnb_config_kwargs
cfg.chat_template
cfg.conversation
cfg.cosine_min_lr_ratio
cfg.dataloader_drop_last
cfg.dataloader_num_workers
cfg.dataloader_pin_memory
cfg.dataloader_prefetch_factor
cfg.dataset_keep_in_memory
cfg.dataset_prepared_path
cfg.dataset_processes
cfg.dataset_shard_idx
cfg.dataset_shard_num
cfg.datasets
cfg.ddp
cfg.ddp_broadcast_buffers
cfg.ddp_bucket_cap_mb
cfg.ddp_timeout
cfg.debug
cfg.deepspeed
cfg.default_system_message
cfg.device
cfg.device_map
cfg.do_bench_eval
cfg.dpo_beta
cfg.dpo_label_smoothing
cfg.eager_attention
cfg.early_stopping_patience
cfg.eval_batch_size
cfg.eval_sample_packing
cfg.eval_steps
cfg.eval_table_max_new_tokens
cfg.eval_table_size
cfg.evals_per_epoch
cfg.evaluation_strategy
cfg.field_input
cfg.field_instruction
cfg.field_output
cfg.field_system
cfg.flash_attention
cfg.flash_attn_cross_entropy
cfg.flash_attn_fuse_mlp
cfg.flash_attn_fuse_qkv
cfg.flash_attn_rms_norm
cfg.flash_optimum
cfg.float16
cfg.format
cfg.fp16
cfg.fsdp
cfg.fsdp_config
cfg.gptq
cfg.gptq_disable_exllama
cfg.gpu_memory_limit
cfg.gradient_accumulation_steps
cfg.gradient_checkpointing
cfg.gradient_checkpointing_kwargs
cfg.greater_is_better
cfg.group_by_length
cfg.hf_use_auth_token
cfg.hub_model_id
cfg.hub_strategy
cfg.is_falcon_derived_model
cfg.is_file
cfg.is_llama_derived_model
cfg.is_mistral_derived_model
cfg.is_preprocess
cfg.is_qwen_derived_model
cfg.learning_rate
cfg.load_best_model_at_end
cfg.load_in_4bit
cfg.load_in_8bit
cfg.local_rank
cfg.logging_steps
cfg.lora_alpha
cfg.lora_dropout
cfg.lora_fan_in_fan_out
cfg.lora_model_dir
cfg.lora_modules_to_save
cfg.lora_on_cpu
cfg.lora_r
cfg.lora_target_linear
cfg.lora_target_modules
cfg.loss_watchdog_patience
cfg.loss_watchdog_threshold
cfg.lr_quadratic_warmup
cfg.lr_scheduler
cfg.lr_scheduler_kwargs
cfg.max_grad_norm
cfg.max_memory
cfg.max_packed_sequence_len
cfg.max_steps
cfg.merge_lora
cfg.metric_for_best_model
cfg.micro_batch_size
cfg.mlflow_experiment_name
cfg.model_config
cfg.model_config_type
cfg.model_kwargs
cfg.model_revision
cfg.model_type
cfg.neftune_noise_alpha
cfg.no_input_format
cfg.noisy_embedding_alpha
cfg.num_epochs
cfg.optimizer
cfg.output_dir
cfg.pad_to_sequence_len
cfg.path
cfg.peft
cfg.peft_adapter
cfg.peft_layers_to_transform
cfg.precompute_ref_log_probs
cfg.pretraining_dataset
cfg.push_dataset_to_hub
cfg.push_to_hub_model_id
cfg.read_text
cfg.relora_cpu_offload
cfg.relora_steps
cfg.relora_warmup_steps
cfg.remove_unused_columns
cfg.resize_token_embeddings_to_32x
cfg.resume_from_checkpoint
cfg.rl
cfg.rl_adapter_ref_model
cfg.rope_scaling
cfg.s2_attention
cfg.sample_packing
cfg.sample_packing_eff_est
cfg.save_safetensors
cfg.save_steps
cfg.save_strategy
cfg.save_total_limit
cfg.saves_per_epoch
cfg.sdp_attention
cfg.seed
cfg.sequence_len
cfg.special_tokens
cfg.strict
cfg.system_format
cfg.system_prompt
cfg.test_datasets
cfg.tf32
cfg.tokenizer_config
cfg.tokenizer_legacy
cfg.tokenizer_type
cfg.tokenizer_use_fast
cfg.tokens
cfg.torch_compile
cfg.torch_compile_backend
cfg.torch_dtype
cfg.torchdistx_path
cfg.total_num_tokens
cfg.total_supervised_tokens
cfg.train_on_inputs
cfg.trust_remote_code
cfg.type
cfg.unfrozen_parameters
cfg.use_mlflow
cfg.use_wandb
cfg.val_set_size
cfg.wandb_name
cfg.wandb_project
cfg.wandb_run_id
cfg.warmup_ratio
cfg.warmup_steps
cfg.weight_decay
cfg.world_size
cfg.xformers_attention
cfg.zero_optimization
```
</details>

## LLM Finetuning with Truefoundry
Test QLoRA w/ Deepspeed Stage 2

```
#!/bin/bash

# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=INFO
# export TORCH_PER_PROCESS_MEMORY_LIMIT=22000
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
# MODEL_ID=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
# MODEL_ID=cognitivecomputations/Wizard-Vicuna-30B-Uncensored
# MODEL_ID=EleutherAI/pythia-70m
MODEL_ID=NousResearch/Llama-2-7b-chat-hf
# MODEL_ID=NousResearch/Llama-2-13b-chat-hf
# MODEL_ID=mistralai/Mistral-7B-Instruct-v0.2
# MODEL_ID=NousResearch/Llama-2-70b-chat-hf
# MODEL_ID=mistralai/Mixtral-8x7B-Instruct-v0.1
# MODEL_ID=stas/tiny-random-llama-2
# MODEL_ID=microsoft/phi-1_5
# MODEL_ID=microsoft/phi-2
# MODEL_ID=Deci/DeciLM-7B
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

#### Experimental things we want to try

- Memory Savings Optimizers
    - AnyPrecision Adam: `--optim adamw_anyprecision --optim-args "use_kahan_summation=True, momentum_dtype=bfloat16, variance_dtype=bfloat16"`
    - 8-bit Adam: `--optim adamw_bnb_8bit`
    - Zero's BF16 optimizer
- torch.compile -> Works in some cases, can speedup training
- Zero++ quantized weights and gradients for faster comm
- Long context
    - Sequence Parallelism w/ Deepspeed Ulysses
    - LongLora with SSA
    - Tricks mentioned in Meta: Effective Long-Context Scaling of Foundation Model
    - Quantized Activations? - FP8 training is already a thing
        - https://github.com/kaiokendev/alpaca_lora_4bit
- DP + TP + PP aka Megatron
    - Difficult to configure, Megatron-Deepspeed provides lower throughput but easier to work with
