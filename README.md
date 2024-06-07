> [!important]
> Please prefer using commits from [release tags](https://github.com/truefoundry/llm-finetune/releases). `main` branch is work in progress and may have partially working commits.

## LLM Finetuning with Truefoundry

Test QLoRA w/ Deepspeed Stage 2

```
./sample_run.sh
```

---

TODO:

- [ ] Setup C/I Tests
- [ ] Track and publish VRAM and Speed benchmarks for popular models and GPUs

---

Generally we always try to optimize for memory footprint because that allows higher batch size and more gpu utilization
Speedup is second priority but we take what we can easily get
