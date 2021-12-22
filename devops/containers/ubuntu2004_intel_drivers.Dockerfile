ARG base_tag=latest
ARG base_image=ghcr.io/intel/llvm/ubuntu2004_base

FROM $base_image:$base_tag

ENV DEBIAN_FRONTEND=noninteractive

ARG compute_runtime_tag=latest
ARG igc_tag=latest
ARG cm_tag=latest
ARG tbb_tag=latest
ARG fpgaemu_tag=latest
ARG cpu_tag=latest

RUN apt update && apt install -yqq wget

COPY scripts/get_release.py /

# Install IGC, CM and NEO
RUN python3 /get_release.py intel/intel-graphics-compiler $igc_tag \
  | grep ".*deb" \
  | wget -qi - && \
  python3 /get_release.py intel/compute-runtime $compute_runtime_tag \
  | grep -E ".*((deb)|(sum))" \
  | wget -qi - && \
  sha256sum -c *.sum &&\
  python3 /get_release.py intel/cm-compiler $cm_tag \
  | grep ".*deb" \
  | wget -qi - && \
  dpkg -i *.deb && rm *.deb *.sum

RUN mkdir /runtimes

# Install TBB
RUN cd /runtimes && \
  python3 /get_release.py oneapi-src/onetbb $tbb_tag \
  | grep -E ".*-lin.tgz" \
  | wget -qi - && \
  tar -xf *.tgz && rm *.tgz && mv oneapi-tbb-* oneapi-tbb

# Install Intel FPGA Emulator
RUN cd /runtimes && \
  python3 /get_release.py intel/llvm $fpgaemu_tag \
  | grep -E ".*fpgaemu.*tar.gz" \
  | wget -qi - && \
  mkdir fpgaemu && tar -xf *.tar.gz -C fpgaemu && rm *.tar.gz && \
  if [ -e /runtimes/fpgaemu/install.sh ]; then \
    bash -x /runtimes/fpgaemu/install.sh ; \
  else \
    echo  /runtimes/fpgaemu/x64/libintelocl_emu.so >  /etc/OpenCL/vendors/intel_fpgaemu.icd ; \
  fi

# Install Intel OpenCL CPU Runtime
RUN cd /runtimes && \
  python3 /get_release.py intel/llvm $cpu_tag \
  | grep -E ".*oclcpuexp.*tar.gz" \
  | wget -qi - && \
  mkdir oclcpu && tar -xf *.tar.gz -C oclcpu && rm *.tar.gz && \
  if [ -e /runtimes/oclcpu/install.sh ]; then \
    bash -x /runtimes/oclcpu/install.sh ; \
  else \
    echo  /runtimes/oclcpu/x64/libintelocl.so > /etc/OpenCL/vendors/intel_oclcpu.icd  ; \
  fi

COPY scripts/drivers_entrypoint.sh /drivers_entrypoint.sh

ENTRYPOINT ["/bin/bash", "/drivers_entrypoint.sh"]

