ARG base_tag=latest
ARG base_image=ghcr.io/intel/llvm/ubuntu2004_intel_drivers

FROM $base_image:$base_tag

COPY scripts/drivers_entrypoint.sh /drivers_entrypoint.sh
ADD llvm_sycl.tar.xz /usr

ENTRYPOINT ["/bin/bash", "/drivers_entrypoint.sh"]

