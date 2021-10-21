FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -yqq \
      build-essential \
      cmake \
      ninja-build \
      ccache \
      git \
      python3
