#!/bin/bash

set -e
set -x
set -o pipefail

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
else
    CR_TAG=$compute_runtime_tag
    IGC_TAG=$igc_tag
    CM_TAG=$cm_tag
    L0_TAG=$level_zero_tag
    TBB_TAG=$tbb_tag
    FPGA_TAG=$fpgaemu_tag
    CPU_TAG=$cpu_tag
fi

function get_release() {
    REPO=$1
    TAG=$2
    URL="https://api.github.com/repos/${REPO}/releases/tags/${TAG}"
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
    if [ -d "$INSTALL_LOCATION/oneapi-tbb" ]; then
      echo "$INSTALL_LOCATION/oneapi-tbb exists and will be removed!"
      rm -Rf $INSTALL_LOCATION/oneapi-tbb;
    fi
    echo "Installing TBB..."
    echo "TBB version $TBB_TAG"
    get_release oneapi-src/onetbb $TBB_TAG \
      | grep -E ".*-lin.tgz" \
      | wget -qi -
    tar -xf *.tgz && rm *.tgz && mv oneapi-tbb-* oneapi-tbb

    TBB_INSTALLED=true
  fi
}

InstallIGFX () {
  echo "Installing Intel Graphics driver..."
  echo "Compute Runtime version $CR_TAG"
  echo "CM compiler version $CM_TAG"
  echo "Level Zero version $L0_TAG"
  echo "IGC version $IGC_TAG"
  UBUNTU_VER="u24\.04"
  get_release intel/intel-graphics-compiler $IGC_TAG \
    | grep ".*deb" \
    | wget -qi -
  get_release intel/compute-runtime $CR_TAG \
    | grep -E ".*((\.deb)|(sum))" \
    | wget -qi -
  # We don't download .ddeb packages, so ignore missing ones.
  sha256sum -c *.sum --ignore-missing
  get_release intel/cm-compiler $CM_TAG \
    | grep ".*deb" \
    | grep -v "u18" \
    | wget -qi -
  get_release oneapi-src/level-zero $L0_TAG \
    | grep ".*$UBUNTU_VER.*deb$" \
    | wget -qi -
  dpkg -i --force-all *.deb && rm *.deb *.sum
  mkdir -p /usr/local/lib/igc/
  echo "$IGC_TAG" > /usr/local/lib/igc/IGCTAG.txt
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

if [[ $# -eq 0 ]] ; then
  echo "No options were specified. Please, specify one or more of the following:"
  echo "--all      - Install all Intel drivers"
  echo "--igfx     - Install Intel Graphics drivers"
  echo "--cpu      - Install Intel CPU OpenCL runtime"
  echo "Set INSTALL_LOCATION env variable to specify install location"
  exit 0
fi

IGCTAG=${IGC_TAG}

while [ "${1:-}" != "" ]; do
  case "$1" in
    "--all")
      InstallIGFX
      InstallTBB
      InstallCPURT
      ;;
    "--igfx")
      InstallIGFX
      ;;
    "--cpu")
      InstallTBB
      InstallCPURT
      ;;
  esac
  shift
done
