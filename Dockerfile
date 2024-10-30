# https://hub.docker.com/layers/winglian/axolotl/main-py3.11-cu121-2.3.1/images/sha256-3cc7799257eb808b819e1170c5781a4e5fe4a457687242d1f36d25fc9d3e98e0?context=explore
FROM winglian/axolotl@sha256:0bdfe3f9b0b4be55a2f74748eff85a16c63558d2cd4c2d4bbbe9081da1ec5640
USER root
COPY requirements.txt /tmp/
RUN pip install -U pip wheel setuptools && \
    pip uninstall -y axolotl && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt
RUN mkdir -p /packages && \
    cd /packages && \
    git clone https://github.com/truefoundry/axolotl && \
    cd axolotl/ && \
    git checkout 811833d0f8ae2123dbe1a0aeb9647f6de36a77ab
RUN cd /packages/axolotl/ && \
    MAX_JOBS=1 NVCC_APPEND_FLAGS="--threads 1" pip install -U --no-build-isolation --no-cache-dir -e .[flash-attn,mamba-ssm,fused-dense-lib,optimizers,lion-pytorch,galore] && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt && \
    rm -rf /root/.cache/pip
COPY plugins/axolotl_truefoundry /packages/axolotl_truefoundry
RUN cd /packages/axolotl_truefoundry/ && \
    pip install --no-cache-dir -e . && \
    rm -rf /root/.cache/pip
WORKDIR /app
COPY . /app
