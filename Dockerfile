# https://hub.docker.com/layers/winglian/axolotl/main-20240819-py3.11-cu121-2.3.1/images/sha256-eb331da0d83e0e55301c542852ee4939d36fa02810f57d99b15f56e4dc0e200d?context=explore
FROM winglian/axolotl@sha256:e70e7ea55ab3ae3c212066bf45271f49198445ab646b9d470d9e0f41050ac8c9
USER root
COPY requirements.txt /tmp/
RUN pip install -U pip wheel setuptools && \
    pip uninstall -y mlflow axolotl && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt
RUN mkdir -p /packages && \
    cd /packages && \
    git clone https://github.com/truefoundry/axolotl && \
    cd axolotl/ && \
    git checkout 294e9097e2c4ea642198aea5ad0561d3b647e572
RUN cd /packages/axolotl/ && \
    MAX_JOBS=1 NVCC_APPEND_FLAGS="--threads 1" pip install -U --no-build-isolation -e .[flash-attn,mamba-ssm,fused-dense-lib,optimizers] && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt && \
    rm -rf /root/.cache/pip
WORKDIR /app
COPY . /app
