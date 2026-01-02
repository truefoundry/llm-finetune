# LLM Finetuning Project

> [!NOTE]
> **Status**: Active Development. Supports Linux (Production) and Mac (Development/Fallback).

This project provides a robust pipeline for fine-tuning Large Language Models (LLMs) using **Axolotl**, **DeepSpeed**, and **Unsloth**. It is designed to be highly configurable and efficient, supporting the latest optimization techniques like QLoRA and Flash Attention 2.

## 🚀 Key Features

-   **High Performance**: Leverages `unsloth` and `flash-attention` for faster training on NVIDIA GPUs.
-   **Multi-Platform Support**: functionality on **Linux** (Production) and **macOS** (Development via Fallback Compatibility Layer).
-   **Experiment Tracking**: Integrated with **TrueFoundry** and **TensorBoard** for metrics and model management.
-   **Modular Configuration**: Easy-to-use YAML configuration for models, datasets, and hyperparameters.

---

## 🛠️ Installation

### 1. Linux (Production / NVIDIA GPU)
This is the recommended way to run full fine-tuning jobs.

**Prerequisites:**
-   Linux OS
-   NVIDIA GPU with CUDA 12.x
-   Docker (Optional but recommended)

**Using Docker (Recommended):**
```bash
docker build -t llm-finetune .
docker run --gpus all -it llm-finetune
```

**Local Install:**
```bash
pip install -r base-requirements.txt
pip install -r requirements.txt
```

### 2. macOS (Apple Silicon / Development)
Run the project on your Mac for development, debugging, and testing the pipeline.
*Note: Advanced optimizations (Unsloth, DeepSpeed sharding) are disabled in this mode.*

**Setup:**
```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Mac-compatible dependencies
pip install -r requirements-mac.txt
```

---

## 🏃 Usage

### Quick Start (Dry Run)
Verify your setup by running a small dry-run with GPT-2.

```bash
# Linux/Mac
python train.py config-base.yaml \
  --base_model gpt2 \
  --output_dir ./outputs \
  --max_steps 1 \
  --train_data_uri ./sample_data/multiply-1k.jsonl
```

### Full Training
Configure your training in `config-base.yaml` or pass arguments via CLI.

```bash
python train.py config-base.yaml \
  --base_model unsloth/Llama-3.2-1B-Instruct \
  --num_epochs 3 \
  --learning_rate 2e-4
```

---

## 📂 Project Structure

-   `train.py`: Main entry point. Contains the training loop and Mac compatibility layer.
-   `utils.py`: Utility functions for memory management and GPU metrics.
-   `config-base.yaml`: Base configuration file for Axolotl.
-   `requirements.txt`: Production dependencies (Linux/CUDA).
-   `requirements-mac.txt`: Development dependencies (Mac/MPS).
-   `Dockerfile`: Container definition for reproducible builds.

---

## 🤝 Contributing

Please ensure you test your changes on both Linux (if possible) and Mac environments.
-   **Mac Users**: Use `requirements-mac.txt`.
-   **Linux Users**: Use `requirements.txt`.

---
*Powered by [TrueFoundry](https://truefoundry.com/)*
