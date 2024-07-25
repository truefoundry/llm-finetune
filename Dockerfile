# https://hub.docker.com/layers/winglian/axolotl-cloud/main-20240725-py3.11-cu121-2.3.1/images/sha256-2a982558e2ba91409d33327c1eee54869da21af62b09ab258585cae2309c6044?context=explore
FROM --platform=linux/amd64 winglian/axolotl@sha256:085b228dc7c493fd0cfad764ea4aeef10b0ca61cf8193944660bb6be8b5160e3
USER root
COPY requirements.txt /tmp/
RUN pip install -U pip wheel setuptools && \
    pip uninstall -y mlflow axolotl && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt
RUN mkdir -p /packages && \
    cd /packages && \
    git clone https://github.com/truefoundry/axolotl && \
    cd axolotl/ && \
    git checkout f16448e84b6135a377512eca79c306d5acf035f6
RUN cd /packages/axolotl/ && \
    MAX_JOBS=1 NVCC_APPEND_FLAGS="--threads 1" pip install -U --no-build-isolation -e .[flash-attn,mamba-ssm,fused-dense-lib,optimizers] && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt && \
    rm -rf /root/.cache/pip
WORKDIR /app
COPY . /app
