#!/bin/bash

set -e
set -x

if [ -f "$1" ]; then
    # Read data from the dependencies.json passed as the first argument.
    CONFIG_FILE=$1
    CR_TAG=$(jq -r '.linux.compute_runtime.github_tag' $CONFIG_FILE)
    IGC_TAG=$(jq -r '.linux.igc.github_tag' $CONFIG_FILE)
    CM_TAG=$(jq -r '.linux.cm.github_tag' $CONFIG_FILE)
    L0_TAG=$(jq -r '.linux.level_zero.github_tag' $CONFIG_FILE)
    TBB_TAG=$(jq -r '.linux.tbb.github_tag' $CONFIG_FILE)
    FPGA_TAG=$(jq -r '.linux.fpgaemu.github_tag' $CONFIG_FILE)
    CPU_TAG=$(jq -r '.linux.oclcpu.github_tag' $CONFIG_FILE)
    if [[ "$*" == *"--use-dev-igc"* ]]; then
       CONFIG_FILE_IGC_DEV=$2
       IGC_DEV_TAG=$(jq -r '.linux.igc_dev.github_tag' $CONFIG_FILE_IGC_DEV)
       IGC_DEV_VER=$(jq -r '.linux.igc_dev.version' $CONFIG_FILE_IGC_DEV)
       IGC_DEV_URL=$(jq -r '.linux.igc_dev.url' $CONFIG_FILE_IGC_DEV)
    fi
elif [[ "$*" == *"--use-latest"* ]]; then
    CR_TAG=latest
    IGC_TAG=latest
    CM_TAG=latest
    L0_TAG=latest
    TBB_TAG=latest
    FPGA_TAG=latest
    CPU_TAG=latest
else
    CR_TAG=$compute_runtime_tag
    IGC_TAG=$igc_tag
    IGC_DEV_TAG=$igc_dev_tag
    IGC_DEV_VER=$igc_dev_ver
    IGC_DEV_URL=$igc_dev_url
    CM_TAG=$cm_tag
    L0_TAG=$level_zero_tag
    TBB_TAG=$tbb_tag
    FPGA_TAG=$fpgaemu_tag
    CPU_TAG=$cpu_tag
fi

function get_release() {
    REPO=$1
    TAG=$2
    if [ "$TAG" == "latest" ]; then
        URL="https://api.github.com/repos/${REPO}/releases/latest"
    else
        URL="https://api.github.com/repos/${REPO}/releases/tags/${TAG}"
    fi
    HEADER=""
    if [ "$GITHUB_TOKEN" != "" ]; then
        HEADER="Authorization: Bearer $GITHUB_TOKEN"
    fi
    curl -s -L -H "$HEADER" $URL \
        | jq -r '. as $raw | try .assets[].browser_download_url catch error($raw)'
}

function get_pre_release_igfx() {
    URL=$1
    HASH=$2
    HEADER=""
    if [ "$GITHUB_TOKEN" != "" ]; then
       HEADER="Authorization: Bearer $GITHUB_TOKEN"
    fi
    curl -L -H "$HEADER" -H "Accept: application/vnd.github.v3+json" $URL -o $HASH.zip
    unzip $HASH.zip && rm $HASH.zip
}

TBB_INSTALLED=false

if [[ -v INSTALL_LOCATION ]]; then
  INSTALL_LOCATION=$(realpath "$INSTALL_LOCATION")
  echo "Installing to $INSTALL_LOCATION"
else
  INSTALL_LOCATION="/opt/runtimes/"
  echo "Install location not specified. Installing to $INSTALL_LOCATION"
fi;

InstallTBB () {
  if [ "$TBB_INSTALLED" = false ]; then
    mkdir -p $INSTALL_LOCATION
    cd $INSTALL_LOCATION
    echo "Installing TBB..."
    echo "TBB version $TBB_TAG"
    get_release oneapi-src/onetbb $TBB_TAG \
      | grep -E ".*-lin.tgz" \
      | wget -qi -
    tar -xf *.tgz && rm *.tgz && mv oneapi-tbb-* oneapi-tbb

    TBB_INSTALLED=true
  fi
}

CheckIGCdevTag() {
    local prefix="igc-dev-"
    local arg="$1"

    if [[ $arg == "$prefix"* ]]; then
       echo "Yes"
    else
       echo "No"
    fi
}

InstallIGFX () {
  echo "Installing Intel Graphics driver..."
  echo "Compute Runtime version $CR_TAG"
  echo "CM compiler version $CM_TAG"
  echo "Level Zero version $L0_TAG"
  IS_IGC_DEV=$(CheckIGCdevTag $IGCTAG)
  IGNORE_CHECKSUM=false
  DPKG_OPTIONS=""
  if [ "$IS_IGC_DEV" == "Yes" ]; then
    echo "IGC dev git hash $IGC_DEV_VER"
    get_pre_release_igfx $IGC_DEV_URL $IGC_DEV_VER
    IGNORE_CHECKSUM=true
    DPKG_OPTIONS=" --force-depends-version"
  else
    echo "IGC version $IGC_TAG"
    get_release intel/intel-graphics-compiler $IGC_TAG \
      | grep ".*deb" \
      | wget -qi -
  fi
  get_release intel/compute-runtime $CR_TAG \
    | grep -E ".*((deb)|(sum))" \
    | wget -qi -
  # Perform the checksum conditionally and then get the release
  ( [ "$IGNORE_CHECKSUM" ] || sha256sum -c *.sum ) && \
  get_release intel/cm-compiler $CM_TAG \
    | grep ".*deb" \
    | grep -v "u18" \
    | wget -qi -
  get_release oneapi-src/level-zero $L0_TAG \
    | grep ".*deb" \
    | wget -qi -
  dpkg -i $DPKG_OPTIONS *.deb && rm *.deb *.sum
}

InstallCPURT () {
  echo "Installing Intel OpenCL CPU Runtime..."
  echo "CPU Runtime version $CPU_TAG"
  mkdir -p $INSTALL_LOCATION
  cd $INSTALL_LOCATION
  if [ -d "$INSTALL_LOCATION/oclcpu" ]; then
    echo "$INSTALL_LOCATION/oclcpu exists and will be removed!"
    rm -Rf $INSTALL_LOCATION/oclcpu;
  fi
  get_release intel/llvm $CPU_TAG \
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
  echo "FPGA Emulator version $FPGA_TAG"
  mkdir -p $INSTALL_LOCATION
  cd $INSTALL_LOCATION
  if [ -d "$INSTALL_LOCATION/fpgaemu" ]; then
    echo "$INSTALL_LOCATION/fpgaemu exists and will be removed!"
    rm -Rf $INSTALL_LOCATION/fpgaemu;
  fi
  get_release intel/llvm $FPGA_TAG \
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
  echo "--use-dev-igc     - Install development version of Intel Graphics drivers instead"
  echo "--cpu      - Install Intel CPU OpenCL runtime"
  echo "--fpga-emu - Install Intel FPGA Fast emulator"
  echo "--use-latest      - Use latest for all tags"
  echo "Set INSTALL_LOCATION env variable to specify install location"
  exit 0
fi

if [[ "$*" == *"--use-dev-igc"* ]]
then
   IGCTAG=${IGC_DEV_TAG}
else
   IGCTAG=${IGC_TAG}
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
