#!/bin/bash

apt update && apt install -yqq \
      build-essential \
      cmake \
      ninja-build \
      ccache \
      git \
      python3 \
      python3-distutils \
      python-is-python3 \
      python3-pip \
      zstd \
      ocl-icd-opencl-dev \
      vim \
      libffi-dev \
      libva-dev \
      libtool \
      wget \
      sudo

pip3 install psutil

