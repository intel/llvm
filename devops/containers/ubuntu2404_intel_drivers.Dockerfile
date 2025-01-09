ARG base_tag=latest
ARG base_image=ghcr.io/intel/llvm/ubuntu2404_base

FROM $base_image:$base_tag

ENV DEBIAN_FRONTEND=noninteractive

ARG use_unstable_driver=true

USER root

RUN apt update && apt install -yqq wget

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

COPY scripts/drivers_entrypoint.sh /drivers_entrypoint.sh

USER sycl_ci

ENTRYPOINT ["/bin/bash", "/drivers_entrypoint.sh"]

