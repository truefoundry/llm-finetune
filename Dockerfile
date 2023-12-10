FROM --platform=linux/amd64 mambaorg/micromamba:1.5.3-jammy
USER root
RUN apt update && apt install -y git gcc g++ && rm -rf /var/lib/apt/lists/*
RUN micromamba install -y -c "nvidia/label/cuda-11.8.0" cuda-nvcc cuda-libraries-dev -n base && \
    micromamba install -y -c "conda-forge" python=3.10 -n base && \
    micromamba clean -y --all && \
    micromamba clean -y --force-pkgs-dirs
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0+PTX"
COPY requirements.txt /tmp/requirements.txt
RUN pip install -U pip wheel setuptools && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt && \
    pip install --no-cache-dir --no-build-isolation -U flash-attn==2.3.6
# Hack to make deepspeed compile ops correctly :/ 
RUN ln -s /opt/conda/lib /opt/conda/lib64
# This is a hacky work around! Ideal way is to use
# /usr/local/bin/_entrypoint.sh as the main entrypoint (command in K8s)
ENV PATH "$MAMBA_ROOT_PREFIX/bin:$PATH"
WORKDIR /app
COPY . /app