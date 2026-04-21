ARG base_tag=latest
ARG base_image=ghcr.io/intel/llvm/ubuntu2404_intel_drivers

FROM $base_image:$base_tag

ENV DEBIAN_FRONTEND=noninteractive

USER root

COPY scripts/download_blender.sh /download_blender.sh
RUN /download_blender.sh

USER sycl

ENTRYPOINT ["/bin/bash", "/drivers_entrypoint.sh"]
