ARG base_tag=latest
ARG base_image=ghcr.io/intel/llvm/ubuntu2004_intel_drivers

FROM $base_image:$base_tag

COPY scripts/drivers_entrypoint.sh /drivers_entrypoint.sh
RUN mkdir -p /opt/sycl
ADD llvm_sycl.tar.zst /tmp
RUN tar -I 'zstd -d' -xf /tmp/llvm_sycl.tar.zst -C /opt/sycl && rm /tmp/llvm_sycl.tar.zst

ENV PATH /opt/sycl/bin:$PATH
ENV LD_LIBRARY_PATH /opt/sycl/lib:$LD_LIBRARY_PATH

ENTRYPOINT ["/bin/bash", "/drivers_entrypoint.sh"]

