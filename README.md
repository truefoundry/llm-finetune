#### Test command

Run with Deepspeed: Pick from one of the deepspeed configs

```shell
deepspeed --num_gpus 1 train.py \
          --output_dir ./model \
          --cleanup_output_dir_on_start \
          --max_num_samples 10 \
          --deepspeed ./1_ds_z1_config.json \
          --half_precision_backend cuda_amp \
          --model_id EleutherAI/pythia-70m \
          --report_to_mlfoundry false \
          --ml_repo transformers \
          --train_data https://assets.production.truefoundry.com/standford_alpaca_train_49k.jsonl \
          --eval_data https://assets.production.truefoundry.com/standford_alpaca_test_2k.jsonl \
          --num_train_epochs 3 \
          --per_device_train_batch_size 4 \
          --per_device_eval_batch_size 4 \
          --learning_rate 0.00005 \
          --warmup_ratio 0.3 \
          --gradient_accumulation_steps 1 \
          --logging_steps 0.1 \
          --logging_strategy steps\
          --seed 42 \
          --data_seed 42 \
          --lr_scheduler_type linear \
          --weight_decay 0.01 \
          --max_grad_norm 1.0 \
          --gradient_checkpointing true
```



#### For CPU


```shell
    python train.py \
                          --output_dir ./model \
                          --cleanup_output_dir_on_start \
                          --max_num_samples 10 \
                          --no_cuda \
                          --model_id EleutherAI/pythia-70m \
                          --report_to_mlfoundry false \
                          --ml_repo transformers \
                          --train_data https://assets.production.truefoundry.com/standford_alpaca_train_49k.jsonl \
                          --eval_data https://assets.production.truefoundry.com/standford_alpaca_test_2k.jsonl \
                          --eval_size 0.1 \
                          --num_train_epochs 3 \
                          --per_device_train_batch_size 4 \
                          --per_device_eval_batch_size 4 \
                          --learning_rate 0.00005 \
                          --warmup_ratio 0.3 \
                          --gradient_accumulation_steps 1 \
                          --logging_steps 0.1 \
                          --logging_strategy steps\
                          --seed 42 \
                          --data_seed 42 \
                          --lr_scheduler_type linear \
                          --weight_decay 0.01 \
                          --max_grad_norm 1.0 \
                          --gradient_checkpointing true
```
# LLM Finetuning with Truefoundry
