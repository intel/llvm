#!/bin/bash

CR_TAG=$compute_runtime_tag
IGC_TAG=$igc_tag
CM_TAG=$cm_tag
TBB_TAG=$tbb_tag
FPGA_TAG=$fpgaemu_tag
CPU_TAG=$cpu_tag

TBB_INSTALLED=false

LOCATION=$(dirname -- "$(readlink -f "${BASH_SOURCE}")")

if [[ -v INSTALL_LOCATION ]]; then
  INSTALL_LOCATION=$(realpath "$INSTALL_LOCATION")
  echo "Installing to $INSTALL_LOCATION"
else
  INSTALL_LOCATION="/opt/runtimes/"
  echo "Install location not specified. Installing to $INSTALL_LOCATION"
fi;

InstallTBB () {
  if [ "$TBB_INSTALLED" = false ]; then
    cd $INSTALL_LOCATION
    echo "Installing TBB..."
    python3 $LOCATION/get_release.py oneapi-src/onetbb $TBB_TAG \
    | grep -E ".*-lin.tgz" \
    | wget -qi - && \
    tar -xf *.tgz && rm *.tgz && mv oneapi-tbb-* oneapi-tbb
  fi
}

InstallIGFX () {
  echo "Installing Intel Graphics driver..."
  python3 $LOCATION/get_release.py intel/intel-graphics-compiler $IGC_TAG \
    | grep ".*deb" \
    | wget -qi -
  python3 $LOCATION/get_release.py intel/compute-runtime $CR_TAG \
    | grep -E ".*((deb)|(sum))" \
    | wget -qi -
  sha256sum -c *.sum && \
  python3 $LOCATION/get_release.py intel/cm-compiler $CM_TAG \
    | grep ".*deb" \
    | wget -qi -
  dpkg -i *.deb && rm *.deb *.sum
}

InstallCPURT () {
  echo "Installing Intel OpenCL CPU Runtime..."
  echo "CPU Runtime version $CPU_TAG"
  cd $INSTALL_LOCATION
  if [ -d "$INSTALL_LOCATION/oclcpu" ]; then
    echo "$INSTALL_LOCATION/oclcpu exists and will be removed!"
    rm -Rf $INSTALL_LOCATION/oclcpu;
  fi
  python3 $LOCATION/get_release.py intel/llvm $CPU_TAG \
    | grep -E ".*oclcpuexp.*tar.gz" \
    | wget -qi -
  mkdir oclcpu && tar -xf *.tar.gz -C oclcpu && rm *.tar.gz
  if [ -e $INSTALL_LOCATION/oclcpu/install.sh ]; then \
    bash -x $INSTALL_LOCATION/oclcpu/install.sh
  else
    echo  $INSTALL_LOCATION/oclcpu/x64/libintelocl.so > /etc/OpenCL/vendors/intel_oclcpu.icd
  fi
}

InstallFPGAEmu () {
  echo "Installing Intel FPGA Fast Emulator..."
  cd $INSTALL_LOCATION
  if [ -d "$INSTALL_LOCATION/fpgaemu" ]; then
    echo "$INSTALL_LOCATION/fpgaemu exists and will be removed!"
    rm -Rf $INSTALL_LOCATION/oclcpu;
  fi
  python3 /get_release.py intel/llvm $FPGA_TAG \
    | grep -E ".*fpgaemu.*tar.gz" \
    | wget -qi - && \
    mkdir fpgaemu && tar -xf *.tar.gz -C fpgaemu && rm *.tar.gz
  if [ -e /runtimes/fpgaemu/install.sh ]; then
    bash -x /runtimes/fpgaemu/install.sh
  else
    echo  /runtimes/fpgaemu/x64/libintelocl_emu.so >  /etc/OpenCL/vendors/intel_fpgaemu.icd
  fi
}

if [[ $# -eq 0 ]] ; then
  echo "No options were specified. Please, specify one or more of the following:"
  echo "--all      - Install all Intel drivers"
  echo "--igfx     - Install Intel Graphics drivers"
  echo "--cpu      - Install Intel CPU OpenCL runtime"
  echo "--fpga-emu - Install Intel FPGA Fast emulator"
  echo "Set INSTALL_LOCATION env variable to specify install location"
  exit 0
fi

while [ "${1:-}" != "" ]; do
  case "$1" in
    "--all")
      InstallIGFX
      InstallTBB
      InstallCPURT
      InstallFPGAEmu
      ;;
    "--igfx")
      InstallIGFX
      ;;
    "--cpu")
      InstallTBB
      InstallCPURT
      ;;
    "--fpga-emu")
      InstallTBB
      InstallFPGAEmu
      ;;
  esac
  shift
done
