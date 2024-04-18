ARG base_tag=latest
ARG base_image=ghcr.io/intel/llvm/ubuntu2204_base

FROM $base_image:$base_tag

ENV DEBIAN_FRONTEND=noninteractive

ARG use_latest=true
ARG use_igc_dev=false

RUN apt update && apt install -yqq wget \
    && if [ "$use_igc_dev" = "true" ]; then apt-get install -yqq libllvm14; fi

COPY scripts/get_release.py /
COPY scripts/install_drivers.sh /
COPY dependencies.json /

RUN mkdir /runtimes
ENV INSTALL_LOCATION=/runtimes
RUN --mount=type=secret,id=github_token \
    if [ "$use_latest" = "true" ]; then \
      install_driver_opt=" --use-latest"; \
    else \
      install_driver_opt=" dependencies.json"; \
      if [ "$use_igc_dev" = "true" ]; then \
        install_driver_opt="$install_driver_opt --use-dev-igc"; \
      fi; \
    fi && \
    GITHUB_TOKEN=$(cat /run/secrets/github_token) /install_drivers.sh $install_driver_opt --all

COPY scripts/drivers_entrypoint.sh /drivers_entrypoint.sh

ENTRYPOINT ["/bin/bash", "/drivers_entrypoint.sh"]

