# https://hub.docker.com/layers/winglian/axolotl/main-20240726-py3.11-cu121-2.3.1/images/sha256-91e79e34ec6955952a96e52b33f6a014447171d452decf7ec75cbc3dd4a39051?context=explore
FROM --platform=linux/amd64 winglian/axolotl@sha256:84429e018ac7b0d46b79de06d6061341bf775a217c64b0dcf54a688e0fe58f1f
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
