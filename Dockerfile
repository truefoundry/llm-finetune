# https://hub.docker.com/layers/winglian/axolotl/main-20240612-py3.11-cu121-2.3.0/images/sha256-798eed818fb11d24a640c0efbf27f65fbaebc1d9a5db210d585aa2a4328e93e1?context=explore
FROM --platform=linux/amd64 winglian/axolotl@sha256:aac52c92ab245793932a635e6dedf14a3a9fb009e40cdf16c10b715f1466afa8
USER root
COPY requirements.txt /tmp/
RUN pip install -U pip wheel setuptools && \
    pip uninstall -y mlflow axolotl && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt
RUN mkdir -p /packages && \
    cd /packages && \
    git clone https://github.com/truefoundry/axolotl && \
    cd axolotl/ && \
    git checkout 0711bfeb6af7d359deb4ee2cae81ceb6890ebf80
RUN cd /packages/axolotl/ && \
    MAX_JOBS=1 NVCC_APPEND_FLAGS="--threads 1" pip install -U --no-build-isolation -e .[flash-attn,mamba-ssm,fused-dense-lib] && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt && \
    rm -rf /root/.cache/pip
WORKDIR /app
COPY . /app
