# https://hub.docker.com/layers/winglian/axolotl/main-py3.11-cu121-2.1.2/images/sha256-93f4859791ce95d591a71b04afceaa68803c205f3c9490ec71539ea53bcebd4f?context=explore
FROM --platform=linux/amd64 winglian/axolotl@sha256:93f4859791ce95d591a71b04afceaa68803c205f3c9490ec71539ea53bcebd4f
USER root
COPY requirements.txt /tmp/
RUN pip install -U pip wheel setuptools && \
    pip uninstall -y mlflow axolotl && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt && \
    mkdir -p /packages && \
    cd /packages && \
    git clone https://github.com/OpenAccess-AI-Collective/axolotl && \
    cd axolotl/ && \
    git checkout ea00dd08528399405943f880b27ecc60e4d83cdf && \
    pip install --no-build-isolation -e .[deepspeed,flash-attn,mamba-ssm,fused-dense-lib] && \
    pip uninstall -y mlflow tfy-mlflow-client && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt
WORKDIR /app
COPY . /app
