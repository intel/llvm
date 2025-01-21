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
      zstd \
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
      jq \
      curl \
      libhwloc-dev \
      libzstd-dev \
      time

# To obtain latest release of spriv-tool.
# Same as what's done in SPRIV-LLVM-TRANSLATOR:
# https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/cec12d6cf46306d0a015e883d5adb5a8200df1c0/.github/workflows/check-out-of-tree-build.yml#L59
. /etc/os-release
curl -L "https://packages.lunarg.com/lunarg-signing-key-pub.asc" | apt-key add -
echo "deb https://packages.lunarg.com/vulkan $VERSION_CODENAME main" | sudo tee -a /etc/apt/sources.list
apt update && apt install -yqq spirv-tools
