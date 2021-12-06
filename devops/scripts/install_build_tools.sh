#!/bin/bash

apt update && apt install -yqq \
      build-essential \
      cmake \
      ninja-build \
      ccache \
      git \
      python3 \
      python3-distutils \
      python-is-python3
