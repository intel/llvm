FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

USER root

# Install SYCL prerequisites
COPY scripts/install_build_tools.sh /install.sh
RUN /install.sh

# libzstd-dev installed by default on Ubuntu 24.04 is not compiled with -fPIC flag.
# This causes linking errors when building SYCL runtime.
# Bug: https://github.com/intel/llvm/issues/15935
# Workaround: build zstd from sources with -fPIC flag.
COPY scripts/build_zstd_1_5_6_ub24.sh /build_zstd_1_5_6_ub24.sh
RUN /build_zstd_1_5_6_ub24.sh

COPY scripts/create-sycl-user.sh /user-setup.sh
RUN /user-setup.sh

COPY actions/cached_checkout /actions/cached_checkout
COPY actions/cleanup /actions/cleanup
COPY scripts/docker_entrypoint.sh /docker_entrypoint.sh
COPY scripts/install_drivers.sh /opt/install_drivers.sh

USER sycl

ENTRYPOINT ["/docker_entrypoint.sh"]
