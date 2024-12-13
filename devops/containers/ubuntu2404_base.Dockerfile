FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

USER root

# Install SYCL prerequisites
COPY scripts/install_build_tools.sh /install.sh
RUN /install.sh

COPY scripts/create-sycl-user.sh /user-setup.sh
RUN --mount=type=secret,id=sycl_passwd /user-setup.sh

COPY actions/cached_checkout /actions/cached_checkout
COPY actions/cleanup /actions/cleanup
COPY scripts/docker_entrypoint.sh /docker_entrypoint.sh
COPY scripts/install_drivers.sh /opt/install_drivers.sh

USER sycl

ENTRYPOINT ["/docker_entrypoint.sh"]
