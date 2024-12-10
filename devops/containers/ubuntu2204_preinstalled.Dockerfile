ARG base_tag=latest
ARG base_image=ghcr.io/intel/llvm/ubuntu2204_intel_drivers

FROM $base_image:$base_tag

COPY scripts/drivers_entrypoint.sh /drivers_entrypoint.sh
RUN mkdir -p /opt/sycl
ADD sycl_linux.tar.gz /opt/sycl/

ENV PATH /opt/sycl/bin:$PATH
ENV LD_LIBRARY_PATH /opt/sycl/lib:$LD_LIBRARY_PATH

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

