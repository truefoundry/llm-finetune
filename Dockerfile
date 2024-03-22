# https://hub.docker.com/layers/winglian/axolotl/main-py3.11-cu121-2.1.2/images/sha256-a794e3d8562d3a9a40296726671480c45951cd6e0ad6e8f359e47e75ccbe22ab?context=explore
FROM --platform=linux/amd64 winglian/axolotl@sha256:dc46cae262116297d23f2b445deda3d4b9759b7da5b318315665036a0e2c7140
USER root
COPY requirements.txt /tmp/
RUN pip install -U pip wheel setuptools && \
    pip uninstall -y mlflow axolotl && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt
RUN mkdir -p /packages && \
    cd /packages && \
    git clone https://github.com/OpenAccess-AI-Collective/axolotl && \
    cd axolotl/ && \
    git checkout 40a88e8c4a2f32b63df0fe2079f7acfe73329273
RUN cd /packages/axolotl/ && \
    MAX_JOBS=1 NVCC_APPEND_FLAGS="--verbose --threads 1" pip install -v -U --no-build-isolation -e .[deepspeed,flash-attn,mamba-ssm,fused-dense-lib] && \
    pip uninstall -y mlflow tfy-mlflow-client && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt && \
    rm -rf /root/.cache/pip
WORKDIR /app
COPY . /app
