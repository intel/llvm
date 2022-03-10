ARG base_tag=latest
ARG base_image=ghcr.io/intel/llvm/ubuntu2004_intel_drivers

FROM $base_image:$base_tag

RUN mkdir -p /opt/sycl
ADD llvm_sycl.tar.xz /opt/sycl

ENV PATH /opt/sycl/bin:$PATH
ENV LD_LIBRARY_PATH /opt/sycl/lib:$LD_LIBRARY_PATH

ENTRYPOINT ["/bin/bash", "/drivers_entrypoint.sh"]

