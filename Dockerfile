# https://hub.docker.com/layers/winglian/axolotl/main-py3.11-cu121-2.1.2/images/sha256-ace38a300833e0e5fb22af7c8692306699ed3614a1f33b37e8bedb1f6ef0ef2e?context=explore
FROM --platform=linux/amd64 winglian/axolotl@sha256:ace38a300833e0e5fb22af7c8692306699ed3614a1f33b37e8bedb1f6ef0ef2e
ARG MAX_JOBS
ARG NVCC_APPEND_FLAGS
USER root
COPY requirements.txt /tmp/
RUN pip install -U pip wheel setuptools && \
    pip uninstall -y mlflow axolotl && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt
RUN mkdir -p /packages && \
    cd /packages && \
    git clone https://github.com/OpenAccess-AI-Collective/axolotl && \
    cd axolotl/ && \
    git checkout 43265208299242e3bc32690e22efadef79365c9d
RUN cd /packages/axolotl/ && \
    MAX_JOBS=${MAX_JOBS} NVCC_APPEND_FLAGS=${NVCC_APPEND_FLAGS} pip install -vvvvvvv -U --no-build-isolation -e .[deepspeed,flash-attn,mamba-ssm,fused-dense-lib] && \
    pip uninstall -y mlflow tfy-mlflow-client && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt && \
    rm -rf /root/.cache/pip
WORKDIR /app
COPY . /app
