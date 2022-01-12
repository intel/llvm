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
      ocl-icd-libopencl1 \
      vim \
      libffi-dev \
      libva-dev \
      libtool      

pip3 install psutil

