FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

USER root

RUN apt update && apt install -yqq \
      build-essential \
      cmake \
      ninja-build \
      ccache \
      git \
      python3 \
      python3-distutils \
      python-is-python3

RUN groupadd -g 1001 sycl && useradd sycl -u 1001 -g 1001 -m -s /bin/bash

COPY scripts/docker_entrypoint.sh /docker_entrypoint.sh

ENTRYPOINT ["/docker_entrypoint.sh"]
