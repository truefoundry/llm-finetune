# https://hub.docker.com/layers/winglian/axolotl/main-20241109-py3.11-cu121-2.3.1/images/sha256-73a9676288cced9762fd093e0c6325731a30b2c55ba9dc324beda67f3ab64612?context=explore
FROM winglian/axolotl@sha256:21a1d9715d9cf3a6d3f37b253bcb1dc378b86b934dcdb25dc3967da57944f7ac
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
    git checkout 285193c1933dac665ae08b9eef95a355117bf8a2 && \
    cd /packages/axolotl/ && \
    MAX_JOBS=1 NVCC_APPEND_FLAGS="--threads 1" pip install -U --no-build-isolation --no-cache-dir -e .[flash-attn,mamba-ssm,optimizers,lion-pytorch,galore] && \
    rm -rf /root/.cache/pip

# Install axolotl_truefoundry plugin with our requirements overrides over axolotl
COPY plugins/axolotl_truefoundry /packages/axolotl_truefoundry
RUN cd /packages/axolotl_truefoundry/ && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt -e . && \
    rm -rf /root/.cache/pip

WORKDIR /app

# Add source code for finetuning
COPY . /app
