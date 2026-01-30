FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

USER root

# Configure LLVM nightly repo
RUN apt-get update -qq && apt-get install --no-install-recommends -yqq curl ca-certificates
RUN curl -sSL https://apt.llvm.org/llvm-snapshot.gpg.key -o /etc/apt/trusted.gpg.d/apt.llvm.org.asc
RUN echo 'deb http://apt.llvm.org/noble/ llvm-toolchain-noble main' > /etc/apt/sources.list.d/llvm.list

# Install SYCL prerequisites
COPY scripts/install_build_tools.sh /install.sh
RUN /install.sh

COPY scripts/install_vulkan.sh /install_vulkan.sh
RUN /install_vulkan.sh

# libzstd-dev installed by default on Ubuntu 24.04 is not compiled with -fPIC flag.
# This causes linking errors when building SYCL runtime.
# Bug: https://github.com/intel/llvm/issues/15935
# Workaround: build zstd from sources with -fPIC flag.
COPY scripts/build_zstd.sh /build_zstd.sh
RUN /build_zstd.sh

SHELL ["/bin/bash", "-ec"]

# Make the directory if it doesn't exist yet.
# This location is recommended by the distribution maintainers.
RUN mkdir --parents --mode=0755 /etc/apt/keyrings
# Download the key, convert the signing-key to a full
# keyring required by apt and store in the keyring directory
RUN wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null && \
# Add rocm repo
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/6.3/ubuntu noble main" \
    | tee /etc/apt/sources.list.d/amdgpu.list && \
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.3 noble main" \
    |  tee --append /etc/apt/sources.list.d/rocm.list && \
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
    | tee /etc/apt/preferences.d/rocm-pin-600 && \
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
    | tee /etc/apt/preferences.d/rocm-pin-600
# Install the ROCM kernel driver
RUN apt update && apt install -yqq rocm-dev && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Fix Vulkan install inside container
# https://stackoverflow.com/questions/74965945/vulkan-is-unable-to-detect-nvidia-gpu-from-within-a-docker-container-when-using
RUN apt-get update && \
    apt-get install -y --no-install-recommends --download-only libnvidia-gl-565 && \
    dpkg-deb --extract /var/cache/apt/archives/libnvidia-gl-565_*.deb extracted && \
    cp -R ./extracted/usr/* /usr/ && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb ./extracted

COPY scripts/create-sycl-user.sh /user-setup.sh
RUN /user-setup.sh

COPY scripts/docker_entrypoint.sh /docker_entrypoint.sh

COPY actions/cached_checkout /actions/cached_checkout
COPY actions/cleanup /actions/cleanup
COPY scripts/docker_entrypoint.sh /docker_entrypoint.sh
COPY scripts/install_drivers.sh /opt/install_drivers.sh

USER sycl

ENTRYPOINT ["/docker_entrypoint.sh"]

