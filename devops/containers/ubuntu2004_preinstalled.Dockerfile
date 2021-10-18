ARG base_tag=latest
ARG base_image=ghcr.io/intel/llvm/ubuntu2004_intel_drivers

FROM $base_image:$base_tag

USER root

COPY scripts/drivers_entrypoint.sh /drivers_entrypoint.sh
ADD llvm_sycl.tar.gz /usr

USER sycl

ENTRYPOINT ["/bin/bash", "/drivers_entrypoint.sh"]

