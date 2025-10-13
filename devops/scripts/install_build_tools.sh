#!/bin/bash

apt update && apt install -yqq \
      build-essential \
      cmake \
      ninja-build \
      ccache \
      git \
      python3 \
      python3-psutil \
      python-is-python3 \
      python3-pip \
      python3-venv \
      ocl-icd-opencl-dev \
      vim \
      libffi-dev \
      libva-dev \
      libtool \
      wget \
      sudo \
      zstd \
      zip \
      unzip \
      pigz \
      jq \
      curl \
      libhwloc-dev \
      libzstd-dev \
      linux-tools-generic \
      linux-tools-common \
      time 

# To obtain latest release of spriv-tool.
# Same as what's done in SPRIV-LLVM-TRANSLATOR:
# https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/cec12d6cf46306d0a015e883d5adb5a8200df1c0/.github/workflows/check-out-of-tree-build.yml#L59
# pkg-config is required for llvm-spriv to detect spriv-tools installation.
. /etc/os-release
curl -L "https://packages.lunarg.com/lunarg-signing-key-pub.asc" | apt-key add -
echo "deb https://packages.lunarg.com/vulkan $VERSION_CODENAME main" | tee -a /etc/apt/sources.list
apt update && apt install -yqq spirv-tools pkg-config

if [[ "$VERSION_CODENAME" == "jammy" ]]; then
    apt-get install -yqq clang-14 libc++-14-dev
else
    apt-get install -yqq clang-18 libc++-18-dev
fi
