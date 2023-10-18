ARG base_tag=latest
ARG base_image=ghcr.io/intel/llvm/ubuntu2204_base

FROM $base_image:$base_tag

ENV DEBIAN_FRONTEND=noninteractive

ARG compute_runtime_tag=latest
ARG igc_tag=latest
ARG cm_tag=latest
ARG level_zero_tag=latest
ARG tbb_tag=latest
ARG fpgaemu_tag=latest
ARG cpu_tag=latest

RUN apt update && apt install -yqq wget

COPY scripts/get_release.py /
COPY scripts/install_drivers.sh /

RUN mkdir /runtimes
ENV INSTALL_LOCATION=/runtimes
RUN --mount=type=secret,id=github_token GITHUB_TOKEN=$(cat /run/secrets/github_token) /install_drivers.sh --all

COPY scripts/drivers_entrypoint.sh /drivers_entrypoint.sh

ENTRYPOINT ["/bin/bash", "/drivers_entrypoint.sh"]

