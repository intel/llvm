FROM ghcr.io/intel/llvm/ubuntu2404_build

ENV DEBIAN_FRONTEND=noninteractive

USER root

# Install oneAPI

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor \
| tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
| tee /etc/apt/sources.list.d/oneAPI.list

# Install the ROCM kernel driver and oneAPI
RUN apt update && apt install -yqq intel-oneapi-compiler-dpcpp-cpp-2025.0 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

USER sycl

ENTRYPOINT ["/docker_entrypoint.sh"]

