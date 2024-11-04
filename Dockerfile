# https://hub.docker.com/layers/winglian/axolotl/main-20241104-py3.11-cu121-2.3.1/images/sha256-790297fa1d71f8f1590c73ca4505ca39fe7dfa2886b2b6862199a6df679bf8e4?context=explore
FROM winglian/axolotl@sha256:cffbcc4993e80301a8918062f8136a6ac402877fd6c29f1168be563e543aee4d
SHELL ["/bin/bash", "-c"]
USER root

# Install torch
COPY requirements.txt /tmp/
RUN pip install -U pip wheel setuptools && \
    pip uninstall -y axolotl && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt

# Install axolotl
RUN mkdir -p /packages && \
    cd /packages && \
    git clone https://github.com/truefoundry/axolotl && \
    cd axolotl/ && \
    git checkout e16f637d079ef5d56321a240ef0547a50c37b708 && \
    cd /packages/axolotl/ && \
    MAX_JOBS=1 NVCC_APPEND_FLAGS="--threads 1" pip install -U --no-build-isolation --no-cache-dir -e .[flash-attn,mamba-ssm,fused-dense-lib,optimizers,lion-pytorch,galore] && \
    rm -rf /root/.cache/pip

# Install axolotl_truefoundry plugin with our requirements overrides over axolotl
COPY plugins/axolotl_truefoundry /packages/axolotl_truefoundry
RUN cd /packages/axolotl_truefoundry/ && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt -e . && \
    rm -rf /root/.cache/pip

WORKDIR /app

# Add source code for finetuning
COPY . /app
