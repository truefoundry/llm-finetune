# https://hub.docker.com/layers/winglian/axolotl/main-20250530/images/sha256-2d4006e9ad7816f3e5b52b5557ccae5ff75d451fc8fb6cd3a6871301d88890fb
FROM winglian/axolotl@sha256:6364ae1f773610b2d4ebb4bfbc04bbd60a3d4dbadbcb16413710cec72f5bbf88
SHELL ["/bin/bash", "-c"]
USER root
RUN [ "$(/usr/local/cuda/bin/nvcc --version | egrep -o "V[0-9]+\.[0-9]+" | cut -c2-)" = "12.4" ] || (echo "Error: CUDA version is not 12.4" && exit 1)

# Install torch and axolotl requirements
COPY base-requirements.txt requirements.txt /tmp/llm-finetune/
RUN pip install -U pip wheel setuptools && \
    pip uninstall -y axolotl torch && \
    pip install -U --no-cache-dir --use-pep517 -r /tmp/llm-finetune/base-requirements.txt && \
    MAX_JOBS=1 NVCC_APPEND_FLAGS="--threads 1" pip install --no-cache-dir --no-build-isolation --use-pep517 -r /tmp/llm-finetune/requirements.txt && \
    rm -rf /root/.cache/pip

# Install axolotl_truefoundry plugin
COPY plugins/axolotl_truefoundry /packages/axolotl_truefoundry
RUN mkdir -p /packages && \
    cd /packages/axolotl_truefoundry/ && \
    pip install --no-cache-dir -e . && \
    rm -rf /root/.cache/pip

WORKDIR /app

# Add source code for finetuning
COPY . /app
