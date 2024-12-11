ARG base_tag=latest
ARG base_image=ghcr.io/intel/llvm/ubuntu2404_base

FROM $base_image:$base_tag

ENV DEBIAN_FRONTEND=noninteractive

ARG use_latest=true

RUN sudo apt update && sudo apt install -yqq wget

COPY scripts/get_release.py /
COPY scripts/install_drivers.sh /
COPY dependencies.json /

RUN sudo mkdir /runtimes
ENV INSTALL_LOCATION=/runtimes
RUN --mount=type=secret,id=github_token \
    if [ "$use_latest" = "true" ]; then \
      install_driver_opt=" --use-latest"; \
    else \
      install_driver_opt=" dependencies.json"; \
    fi && \
    GITHUB_TOKEN=$(cat /run/secrets/github_token) sudo /install_drivers.sh $install_driver_opt --all

COPY scripts/drivers_entrypoint.sh /drivers_entrypoint.sh

USER sycl

ENTRYPOINT ["/bin/bash", "/drivers_entrypoint.sh"]

