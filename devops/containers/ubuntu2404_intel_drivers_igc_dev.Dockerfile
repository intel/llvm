ARG base_tag=latest
ARG base_image=ghcr.io/intel/llvm/ubuntu2404_base

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

# By default Ubuntu sets an arbitrary UID value, that is different from host
# system. When CI passes default UID value of 1001, some of LLVM tools fail to
# discover user home directory and fail a few LIT tests. Fixes UID and GID to
# 1001, that is used as default by GitHub Actions.
RUN groupadd -g 1001 sycl && useradd sycl -u 1001 -g 1001 -m -s /bin/bash
# Add sycl user to video/irc groups so that it can access GPU
RUN usermod -aG video sycl
RUN usermod -aG irc sycl

USER sycl

ENTRYPOINT ["/bin/bash", "/drivers_entrypoint.sh"]

