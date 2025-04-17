FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

USER root

# Install Nvidia keys
# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

# Install SYCL prerequisites
COPY scripts/install_build_tools.sh /install.sh
RUN /install.sh

# Install ROCM

# Make the directory if it doesn't exist yet.
# This location is recommended by the distribution maintainers.
RUN mkdir --parents --mode=0755 /etc/apt/keyrings
# Download the key, convert the signing-key to a full
# keyring required by apt and store in the keyring directory
RUN wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null && \
# Add rocm repo
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.1.1 jammy main" \
| tee --append /etc/apt/sources.list.d/rocm.list && \
printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | tee /etc/apt/preferences.d/rocm-pin-600
# Install the kernel driver
RUN apt update && apt install -yqq rocm-dev && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

COPY llvm/unified-runtime/third_party/requirements.txt /tmp/requirements.txt
RUN apt update && apt install -y python3-venv && \
    python3 -m venv /opt/unified-runtime/.venv && \
    /opt/unified-runtime/.venv/bin/pip install --upgrade pip && \
    /opt/unified-runtime/.venv/bin/pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y intel-oneapi-runtime-opencl intel-oneapi-base-toolkit

COPY scripts/create-sycl-user.sh /user-setup.sh
RUN /user-setup.sh

COPY scripts/docker_entrypoint.sh /docker_entrypoint.sh

USER sycl

ENTRYPOINT ["/docker_entrypoint.sh"]

