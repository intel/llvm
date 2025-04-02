ARG base_tag=latest
ARG base_image=ghcr.io/intel/llvm/ubuntu2404_base

FROM $base_image:$base_tag

ENV DEBIAN_FRONTEND=noninteractive

ARG use_unstable_driver=true

USER root

RUN apt update && apt install -yqq wget ca-certificates gpg

RUN apt-get update && apt --fix-broken install -y

COPY scripts/get_release.py /
COPY scripts/install_drivers.sh /
COPY dependencies.json /

RUN mkdir /runtimes
ENV INSTALL_LOCATION=/runtimes
RUN --mount=type=secret,id=github_token \
    if [ "$use_unstable_driver" = "true" ]; then \
      install_driver_opt=" --use-latest"; \
    else \
      install_driver_opt=" dependencies.json"; \
    fi && \
    GITHUB_TOKEN=$(cat /run/secrets/github_token) /install_drivers.sh $install_driver_opt --all

RUN echo "Installing Intel OpenCL..." && \
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y intel-oneapi-runtime-opencl intel-oneapi-base-toolkit

COPY scripts/drivers_entrypoint.sh /drivers_entrypoint.sh

USER sycl

ENTRYPOINT ["/bin/bash", "/drivers_entrypoint.sh"]

