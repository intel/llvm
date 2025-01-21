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

# Add LLVM's GPG key to obtain latest release of spriv-tool.
# Same as what's done in SPRIV-LLVM-TRANSLATOR:
# https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/cec12d6cf46306d0a015e883d5adb5a8200df1c0/.github/workflows/check-out-of-tree-build.yml#L59
curl -L "https://apt.llvm.org/llvm-snapshot.gpg.key" | sudo apt-key add -
curl -L "https://packages.lunarg.com/lunarg-signing-key-pub.asc" | sudo apt-key add -
echo "deb https://apt.llvm.org/jammy/ llvm-toolchain-jammy main" | sudo tee -a /etc/apt/sources.list
echo "deb https://packages.lunarg.com/vulkan jammy main" | sudo tee -a /etc/apt/sources.list
apt update && apt install -yqq spirv-tools
