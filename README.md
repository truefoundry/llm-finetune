## LLM Finetuning with Truefoundry

Test QLoRA

```
TORCH_PER_PROCESS_MEMORY_LIMIT=0.95 CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
--mixed_precision bf16 \
train.py \
--use_ddp true \
--bf16 true \
--model_id NousResearch/Llama-2-7b-hf \
--use_flash_attention true \
--train_data file:///absolute/path/to/my/data.jsonl \
--max_num_samples 0 \
--eval_data NA \
--eval_size 0.05 \
--num_train_epochs 1 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 4 \
--learning_rate 0.0003 \
--output_dir ./model \
--train_on_prompt false \
--logging_steps 5 \
--logging_strategy steps \
--save_strategy steps \
--save_steps 0.05 \
--evaluation_strategy steps \
--eval_steps 0.05 \
--use_qlora true \
--qlora_bit_length 4 \
--lora_target_modules auto \
--lora_r 32 \
--lora_alpha 64 \
--lora_dropout 0.05 \
--lora_bias none \
--mlfoundry_enable_reporting false \
--mlfoundry_ml_repo my-ml-repo \
--mlfoundry_log_checkpoints true \
--cleanup_output_dir_on_start true
```

- A lot of args are set with defaults in `train.args`
- `TORCH_PER_PROCESS_MEMORY_LIMIT` allows limiting the max memory per gpu. Can be a fraction (denoting percentage) or integer (denoting limit in MiB). Useful for testing limited gpu memory scenarios
- CUDA_VISIBLE_DEVICES can be used to control the amount of GPUs
- `--mixed_precision bf16` and `--bf16 true`. On V100 and T4,  use `--mixed_precision fp16` and `--fp16 true`
- `--use_flash_attention true/false`. This is currently only supported for a select few models like Llama and Mistral/Zephyr. By default this is false
- `--per_device_train_batch_size int` - Because auto batch size finder is enabled, the trainer will try the highest batch size starting from this value, halving it if it does not fit. For e.g. with 8, trainer will try 8, 4, 2, 1 and then crash
- `--mlfoundry_enable_reporting true/false` toggles reporting metrics, checkpoints and models to mlfoundry
- When you are testing locally, you can set `--cleanup_output_dir_on_start true` if you don't care about checkpoints between runs
