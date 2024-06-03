# https://hub.docker.com/layers/winglian/axolotl/main-20240603-py3.11-cu121-2.3.0/images/sha256-e4b898a0f700eb86f9e802bb85c1ec6c509b2dec65d941ad43405fe323865017?context=explore
FROM --platform=linux/amd64 winglian/axolotl@sha256:a66d1469cdad472779f6419ea67d0fbb2cce984244aa86f40c99abaa4a21b3db
USER root
COPY requirements.txt /tmp/
RUN pip install -U pip wheel setuptools && \
    pip uninstall -y mlflow axolotl && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt
RUN mkdir -p /packages && \
    cd /packages && \
    git clone https://github.com/truefoundry/axolotl && \
    cd axolotl/ && \
    git checkout 766b38b15daf62107f2d02d0f6dd3a3a33196581
RUN cd /packages/axolotl/ && \
    MAX_JOBS=1 NVCC_APPEND_FLAGS="--threads 1" pip install -U --no-build-isolation -e .[flash-attn,mamba-ssm,fused-dense-lib] && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt && \
    rm -rf /root/.cache/pip
WORKDIR /app
COPY . /app
