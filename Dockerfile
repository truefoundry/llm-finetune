# https://hub.docker.com/layers/winglian/axolotl/main-20241111-py3.11-cu121-2.3.1/images/sha256-67c35533cf8e7a399de19cdaf3852be093b9e184b9554ea38801a482da5d7231?context=explore
FROM winglian/axolotl@sha256:1f892444717a6781ad0e6e02b3548cd76be14d65a7162f2d82eab5c809936bc5
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
    git checkout b8db5db0fea9a1dad15338c1daf73a04f647caf4 && \
    cd /packages/axolotl/ && \
    MAX_JOBS=1 NVCC_APPEND_FLAGS="--threads 1" pip install -U --use-pep517 --no-build-isolation --no-cache-dir -e .[flash-attn,mamba-ssm,optimizers,lion-pytorch,galore] && \
    rm -rf /root/.cache/pip

# Install axolotl_truefoundry plugin with our requirements overrides over axolotl
COPY plugins/axolotl_truefoundry /packages/axolotl_truefoundry
RUN cd /packages/axolotl_truefoundry/ && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt -e . && \
    rm -rf /root/.cache/pip

WORKDIR /app

# Add source code for finetuning
COPY . /app
