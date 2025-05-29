ARG base_tag=latest
ARG base_image=ghcr.io/intel/llvm/ubuntu2404_base

FROM $base_image:$base_tag

ENV DEBIAN_FRONTEND=noninteractive

USER root

RUN curl -sSL https://apt.llvm.org/llvm-snapshot.gpg.key -o /etc/apt/trusted.gpg.d/apt.llvm.org.asc
RUN echo 'deb http://apt.llvm.org/noble/ llvm-toolchain-noble main' > /etc/apt/sources.list.d/llvm.list

USER sycl

ENTRYPOINT ["/docker_entrypoint.sh"]
