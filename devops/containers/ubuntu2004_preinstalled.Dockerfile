ARG base_tag=latest
ARG base_image=ghcr.io/intel/llvm/ubuntu2004_intel_drivers

FROM $base_image:$base_tag

COPY scripts/drivers_entrypoint.sh /drivers_entrypoint.sh
RUN mkdir -p /opt/sycl
RUN tar -I 'zstd -d' -xf llvm_sycl.tar.zst -C /tmp/llvm_sycl
ADD /tmp/llvm_sycl /opt/sycl

ENV PATH /opt/sycl/bin:$PATH
ENV LD_LIBRARY_PATH /opt/sycl/lib:$LD_LIBRARY_PATH

ENTRYPOINT ["/bin/bash", "/drivers_entrypoint.sh"]

