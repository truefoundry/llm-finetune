# https://hub.docker.com/layers/winglian/axolotl/main-20241217/images/sha256-5ed6e068d193ac35d092f8d6ccb56b1750779415cd07047edbbfb8d4edd87ae2
FROM winglian/axolotl@sha256:0966ba0bdfda0a317016614a6eb9f599325d0e42109544f95f5540d144ddeebd
SHELL ["/bin/bash", "-c"]
USER root
RUN [ "$(/usr/local/cuda/bin/nvcc --version | egrep -o "V[0-9]+\.[0-9]+" | cut -c2-)" = "12.1" ] || (echo "Error: CUDA version is not 12.1" && exit 1)

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
