ARG base_tag=latest
ARG base_image=ghcr.io/intel/llvm/ubuntu2404_intel_drivers

FROM $base_image:$base_tag

ENV DEBIAN_FRONTEND=noninteractive

USER root

COPY scripts/download_blender.sh /download_blender.sh
RUN /download_blender.sh

# Install the GPU ray tracing library which is not part of the standard driver install.
# The official repo provides no binaries, so instead use the intel-graphics PPA build.
# This depends on other components from the GPU driver but we installed them outside
# of the package manager, so use dpkg --force-all to ignore dependency errors.
RUN wget https://launchpadlibrarian.net/847063136/libze-intel-gpu-raytracing_1.2.2-1~24.04~ppa1_amd64.deb -O rt.deb && \
    sudo dpkg --force-all -i rt.deb && \
    rm rt.deb

USER sycl

ENTRYPOINT ["/bin/bash", "/drivers_entrypoint.sh"]
