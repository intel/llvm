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
    if [[ "$*" == *"--use-dev-igc"* ]]; then
       CONFIG_FILE_IGC_DEV=$2
       IGC_DEV_TAG=$(jq -r '.linux.igc_dev.github_tag' $CONFIG_FILE_IGC_DEV)
       IGC_DEV_VER=$(jq -r '.linux.igc_dev.version' $CONFIG_FILE_IGC_DEV)
       IGC_DEV_URL=$(jq -r '.linux.igc_dev.url' $CONFIG_FILE_IGC_DEV)
    fi
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

function build_level_zero_from_source() {
    COMMIT=$1

    apt-get update -qq
    apt-get install -y build-essential cmake git libc6-dev linux-libc-dev

    BUILD_DIR="/tmp/level-zero-build"
    mkdir -p $BUILD_DIR && cd $BUILD_DIR

    git clone https://github.com/oneapi-src/level-zero.git
    
    cd level-zero
    git checkout $COMMIT

    mkdir build && cd build

    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DLEVEL_ZERO_BUILD_TESTS=OFF \
      -DLEVEL_ZERO_BUILD_SAMPLES=OFF

    make -j$(nproc)
    make install

    ldconfig

    rm -rf $BUILD_DIR

    echo "Level Zero built and installed successfully from commit $COMMIT"
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

CheckIGCdevTag() {
    local prefix="igc-dev-"
    local arg="$1"

    if [[ $arg == "$prefix"* ]]; then
       echo "Yes"
    else
       echo "No"
    fi
}

CheckIfCommitHash() {
    local arg="$1"
    if [[ $arg =~ ^[a-f0-9]{40}$ ]]; then
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
  echo "IGC version $IGC_TAG"
  # Always install released igc version first to get rid of the dependency issue
  # by installing the igc first, we will satisfy all the dpkg dependencies .
  # When we install dev igc later, it will then be treated as downgrade (because dev igc come with lowest version 1.0).
  # This can help us avoid using the risky force-depends-version option in dpkg command.
  #
  # Of course, this also installed the libopencl-clang so that we can copy and use later as a temporariy workaround.
  IS_IGC_DEV=$(CheckIGCdevTag $IGCTAG)
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
  
  # Check if L0_TAG is a commit hash or a regular tag
  IS_L0_COMMIT=$(CheckIfCommitHash $L0_TAG)
  if [ "$IS_L0_COMMIT" == "Yes" ]; then
    echo "Level Zero is using commit hash, building from source..."
    if ! build_level_zero_from_source $L0_TAG; then
      exit 1
    fi
    # Install other packages (Level Zero was already installed from source)
    if ls *.deb 1> /dev/null 2>&1; then
      dpkg -i --force-all *.deb && rm *.deb
    fi
    if ls *.sum 1> /dev/null 2>&1; then
      rm *.sum
    fi
  else
    get_release oneapi-src/level-zero $L0_TAG \
      | grep ".*$UBUNTU_VER.*deb$" \
      | wget -qi -
    # Install all packages including Level Zero
    dpkg -i --force-all *.deb && rm *.deb *.sum
  fi
  mkdir -p /usr/local/lib/igc/
  echo "$IGC_TAG" > /usr/local/lib/igc/IGCTAG.txt
  if [ "$IS_IGC_DEV" == "Yes" ]; then
    # Dev IGC deb package did not include libopencl-clang
    # opencl-clang repo does not provide release deb package either.
    # Backup and install it from release igc as a temporarily workaround
    # while we working to resolve the issue.
    echo "Backup libopencl-clang"
    cp -d /usr/local/lib/libopencl-clang2.so.15*  .
    echo "Download IGC dev git hash $IGC_DEV_VER"
    get_pre_release_igfx $IGC_DEV_URL $IGC_DEV_VER
    echo "Install IGC dev git hash $IGC_DEV_VER"
    # New dev IGC packaged iga64 conflicting with iga64 from intel-igc-media
    # force overwrite to workaround it first.
    dpkg -i --force-all *.deb
    echo "Install libopencl-clang"
    # Workaround only, will download deb and install with dpkg once fixed.
    cp -d libopencl-clang2.so.15*  /usr/local/lib/
    rm /usr/local/lib/libigc.so /usr/local/lib/libigc.so.1* && \
       ln -s /usr/local/lib/libigc.so.2 /usr/local/lib/libigc.so && \
       ln -s /usr/local/lib/libigc.so.2 /usr/local/lib/libigc.so.1
    echo "Clean up"
    rm *.deb libopencl-clang2.so.15*
    echo "$IGC_DEV_TAG" > /usr/local/lib/igc/IGCTAG.txt
  fi
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
    mkdir -p /etc/OpenCL/vendors
    echo  $INSTALL_LOCATION/oclcpu/x64/libintelocl.so > /etc/OpenCL/vendors/intel_oclcpu.icd
  fi
}

if [[ $# -eq 0 ]] ; then
  echo "No options were specified. Please, specify one or more of the following:"
  echo "--all      - Install all Intel drivers"
  echo "--igfx     - Install Intel Graphics drivers"
  echo "--use-dev-igc     - Install development version of Intel Graphics drivers instead"
  echo "--cpu      - Install Intel CPU OpenCL runtime"
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
