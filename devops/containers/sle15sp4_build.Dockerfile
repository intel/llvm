FROM registry.suse.com/suse/sle15:15.4

USER root

RUN zypper --non-interactive refresh && \
    zypper --non-interactive install --no-recommends \
      python311 \
      ninja \
      cmake \
      gcc \
      gcc-c++ \
      autoconf \
      automake \
      libtool \
      awk \
      git \
      gzip \
      ccache && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    zypper clean --all

COPY scripts/build_zstd.sh /build_zstd.sh
RUN /build_zstd.sh

COPY scripts/create-sycl-user.sh /user-setup.sh
RUN /user-setup.sh

COPY scripts/docker_entrypoint.sh /docker_entrypoint.sh

USER sycl

ENTRYPOINT ["/docker_entrypoint.sh"]
