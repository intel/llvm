FROM suse/sle15:15.4

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
      awk && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    zypper clean --all

COPY scripts/create-sycl-user.sh /user-setup.sh
RUN /user-setup.sh

COPY scripts/docker_entrypoint.sh /docker_entrypoint.sh

USER sycl

ENTRYPOINT ["/docker_entrypoint.sh"]
