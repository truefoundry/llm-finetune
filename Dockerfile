# https://hub.docker.com/layers/winglian/axolotl/main-20240423-py3.11-cu121-2.2.1/images/sha256-fc2b9d2b1e46d6b7c47c28a65d2c1d2c3ae4f032fafef27ffaf6ec63bf442f44?context=explore
FROM --platform=linux/amd64 winglian/axolotl@sha256:e0b5b8a94934aaf183932c66ab3ce3ad822e91e19341ade8dbf9eccd9339d799
USER root
COPY requirements.txt /tmp/
RUN pip install -U pip wheel setuptools && \
    pip uninstall -y mlflow axolotl && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt
RUN mkdir -p /packages && \
    cd /packages && \
    git clone https://github.com/truefoundry/axolotl && \
    cd axolotl/ && \
    git checkout 7ac62f5fa6b3df526a7d0fed7c711faa20df12b0
RUN cd /packages/axolotl/ && \
    MAX_JOBS=1 NVCC_APPEND_FLAGS="--threads 1" pip install -U --no-build-isolation -e .[flash-attn,mamba-ssm,fused-dense-lib] && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt && \
    rm -rf /root/.cache/pip
WORKDIR /app
COPY . /app
