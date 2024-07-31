ARG base_tag=latest
ARG base_image=ghcr.io/intel/llvm/ubuntu2204_base

FROM $base_image:$base_tag

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -yqq libllvm14

COPY scripts/get_release.py /
COPY scripts/install_drivers.sh /
COPY dependencies.json /
COPY dependencies-igc-dev.json /

RUN mkdir /runtimes
ENV INSTALL_LOCATION=/runtimes
RUN --mount=type=secret,id=github_token \
    install_driver_opt="dependencies.json dependencies-igc-dev.json --use-dev-igc"; \
    GITHUB_TOKEN=$(cat /run/secrets/github_token) /install_drivers.sh $install_driver_opt --all

COPY scripts/drivers_entrypoint.sh /drivers_entrypoint.sh

ENTRYPOINT ["/bin/bash", "/drivers_entrypoint.sh"]

