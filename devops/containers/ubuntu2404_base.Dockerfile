FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

USER root

# Install SYCL prerequisites
COPY scripts/install_build_tools.sh /install.sh
RUN /install.sh

# By default Ubuntu sets an arbitrary UID value, that is different from host
# system. When CI passes default UID value of 1001, some of LLVM tools fail to
# discover user home directory and fail a few LIT tests. Fixes UID and GID to
# 1001, that is used as default by GitHub Actions.
RUN groupadd -g 1001 sycl && useradd sycl -u 1001 -g 1001 -m -s /bin/bash
# Add sycl user to video/irc groups so that it can access GPU
RUN usermod -aG video sycl
RUN usermod -aG irc sycl

# group 109 is required for sycl user to access PVC card.
RUN groupadd -g 109 render
RUN usermod -aG render sycl

# Allow sycl user to run as sudo
RUN echo "sycl  ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

COPY actions/cached_checkout /actions/cached_checkout
COPY actions/cleanup /actions/cleanup
COPY scripts/docker_entrypoint.sh /docker_entrypoint.sh
COPY scripts/install_drivers.sh /opt/install_drivers.sh

USER sycl

ENTRYPOINT ["/docker_entrypoint.sh"]
