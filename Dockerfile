# TODO(chiragjn): Switch to nvcr.io/nvidia/pytorch:23.05-py3 or nvcr.io/nvidia/pytorch:22.12-py3
FROM --platform=linux/amd64 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
WORKDIR /app
RUN apt update && \
    apt install -y git && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN pip install -U pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir --no-build-isolation -U flash-attn==2.3.4
COPY . /app
