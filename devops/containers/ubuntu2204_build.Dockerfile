FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

USER root

# Install Nvidia keys
# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

# Install SYCL prerequisites
COPY scripts/install_build_tools.sh /install.sh
RUN /install.sh

RUN apt install -yqq libnuma-dev wget gnupg2 && \
  apt-get install -yqq gcc-12 g++-12 libstdc++-12-dev && \
  update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 && \
  update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100

RUN wget https://repo.radeon.com/amdgpu-install/6.1/ubuntu/jammy/amdgpu-install_6.1.60100-1_all.deb && \
  apt install -yqq ./amdgpu-install_6.1.60100-1_all.deb && \
  amdgpu-install -y --usecase=rocmdev && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# By default Ubuntu sets an arbitrary UID value, that is different from host
# system. When CI passes default UID value of 1001, some of LLVM tools fail to
# discover user home directory and fail a few LIT tests. Fixes UID and GID to
# 1001, that is used as default by GitHub Actions.
RUN groupadd -g 1001 sycl && useradd sycl -u 1001 -g 1001 -m -s /bin/bash
# Add sycl user to video/irc groups so that it can access GPU
RUN usermod -aG video sycl
RUN usermod -aG irc sycl

COPY scripts/docker_entrypoint.sh /docker_entrypoint.sh

ENTRYPOINT ["/docker_entrypoint.sh"]

