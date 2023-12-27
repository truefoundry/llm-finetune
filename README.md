## LLM Finetuning with Truefoundry

Test QLoRA w/ Deepspeed Stage 2

```
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=INFO
AUTO_FIND_BATCH_SIZE=false
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
LORA_R=32
LORA_ALPHA=64
TORCH_PER_PROCESS_MEMORY_LIMIT=0.95
CUDA_VISIBLE_DEVICES=0,1
TRAIN_DATA="./data.jsonl"
MAX_NUM_SAMPLES=0
MODEL_ID=mistralai/Mistral-7B-Instruct-v0.2
USE_FLASH_ATTENTION=true
GRADIENT_CHECKPOINTING=true
NUM_TRAIN_EPOCHS=3

accelerate launch \
--mixed_precision bf16 \
--use_deepspeed \
train.py \
--deepspeed ./3_ds_z2_config.json \
--bf16 true \
--use_flash_attention $USE_FLASH_ATTENTION \
--model_id $MODEL_ID \
--train_data $TRAIN_DATA \
--max_num_samples $MAX_NUM_SAMPLES \
--eval_data NA \
--eval_size 0.1 \
--num_train_epochs $NUM_TRAIN_EPOCHS \
--gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
--gradient_checkpointing $GRADIENT_CHECKPOINTING \
--learning_rate 0.00001 \
--output_dir ./model \
--train_on_prompt false \
--logging_strategy steps \
--logging_steps 1 \
--save_strategy steps \
--save_steps 0.05 \
--evaluation_strategy steps \
--eval_steps 0.05 \
--use_qlora true \
--lora_target_modules auto \
--lora_r $LORA_R \
--lora_alpha $LORA_ALPHA \
--mlfoundry_enable_reporting false \
--mlfoundry_ml_repo my-ml-repo \
--mlfoundry_run_name test \
--mlfoundry_checkpoint_artifact_name chk-test \
--mlfoundry_log_checkpoints false \
--resume_from_checkpoint false \
--auto_find_batch_size $AUTO_FIND_BATCH_SIZE \
--per_device_train_batch_size $TRAIN_BATCH_SIZE \
--per_device_eval_batch_size $TRAIN_BATCH_SIZE \
--cleanup_output_dir_on_start true \
--torch_compile false
```

To use ddp, remove `--use_deepspeed` and `--deepspeed`

```
accelerate launch \
--mixed_precision bf16 \
train.py \
--use_ddp true \
...
```

- A lot of args are set with defaults in `train.args`
- `TORCH_PER_PROCESS_MEMORY_LIMIT` allows limiting the max memory per gpu. Can be a fraction (denoting percentage) or integer (denoting limit in MiB). Useful for testing limited gpu memory scenarios
- CUDA_VISIBLE_DEVICES can be used to control the amount of GPUs
- `--mixed_precision bf16` and `--bf16 true`. On V100 and T4,  use `--mixed_precision fp16` and `--fp16 true`
- `--use_flash_attention true/false`. This is currently only supported for a select few models like Llama and Mistral/Zephyr. By default this is false
- `--per_device_train_batch_size int` - Because auto batch size finder is enabled, the trainer will try the highest batch size starting from this value, halving it if it does not fit. For e.g. with 8, trainer will try 8, 4, 2, 1 and then crash. Note this only works without Deepspeed!
- `--mlfoundry_enable_reporting true/false` toggles reporting metrics, checkpoints and models to mlfoundry
- When you are testing locally, you can set `--cleanup_output_dir_on_start true` if you don't care about checkpoints between runs

---

Generally we always try to optimize for memory footprint because that allows higher batch size and more gpu utilization
Speedup is second priority but we take what we can easily get

#### Non experimental todo

- Sample Packing
- Dataset Streaming
- IA3 PEFT
- Intelligently patch deepspeed zero arguments that affect memory footprint

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


#### Known issues

- We are running a modified commit of Deepspeed with fix for a memory leak till the PR gets upstreamed
- We are running a non released commit of transformers till > 4.36.2 is released
    - Fixes deepspeed checkpoint size
    - Monkey patched deepspeed checkpoint loading for resuming
- QLoRA cannot work with Zero stage 3 -> Quantized Parameters cannot be sharded at the moment
- Auto batch size finder does not work with Deepspeed
    - There is wrapping bug which does not re-intialize Deepspeed again on new trial. That will be fixed but is unlikely to work well the way auto batch size finder is implemented
- Deepspeed 0.12.4+ FusedAdam has index error bug - Hard to reproduce!
- Some users have reported loss diverges after some time when compared to DDP
- Zero stage 3 ends up giving much worse
