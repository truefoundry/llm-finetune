# https://hub.docker.com/layers/winglian/axolotl/main-py3.11-cu121-2.1.2/images/sha256-1a47b6e45a4e858568e47abe62315bf083441b6561e43ad8684a5ef41cc573a1?context=explore
FROM --platform=linux/amd64 docker pull winglian/axolotl:6508c133725453bc9bb48e13be8542e3b243da552140da3ac06834b4551399aa
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
