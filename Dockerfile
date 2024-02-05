# https://hub.docker.com/layers/winglian/axolotl/main-py3.11-cu121-2.1.2/images/sha256-1a47b6e45a4e858568e47abe62315bf083441b6561e43ad8684a5ef41cc573a1?context=explore
FROM --platform=linux/amd64 docker pull winglian/axolotl:1a47b6e45a4e858568e47abe62315bf083441b6561e43ad8684a5ef41cc573a1
USER root
COPY requirements.txt /tmp/
RUN pip install -U pip wheel setuptools && \
    pip install --no-cache-dir --no-build-isolation -U -r /tmp/requirements.txt
WORKDIR /app
COPY . /app
