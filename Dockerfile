# https://hub.docker.com/layers/winglian/axolotl/main-20240626-py3.11-cu121-2.3.0/images/sha256-d157d1b80bfbbea689e9a4ea233d04bbc37f684f82e01d9dd6730dd0251e61fe?context=explore
FROM --platform=linux/amd64 winglian/axolotl@sha256:7945505e1651a474aa11ed4d70188ff5c5052e17f61bb5f60b956ad8f082328f
USER root
COPY requirements.txt /tmp/
RUN pip install -U pip wheel setuptools && \
    pip uninstall -y mlflow axolotl && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt
RUN mkdir -p /packages && \
    cd /packages && \
    git clone https://github.com/truefoundry/axolotl && \
    cd axolotl/ && \
    git checkout 6a3ca76d3876ba7d9de9480cd203a8356e7279a7
RUN cd /packages/axolotl/ && \
    MAX_JOBS=1 NVCC_APPEND_FLAGS="--threads 1" pip install -U --no-build-isolation -e .[flash-attn,mamba-ssm,fused-dense-lib] && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt && \
    rm -rf /root/.cache/pip
WORKDIR /app
COPY . /app
