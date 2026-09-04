#!/bin/bash

set -x
set -e
set -o pipefail

. /etc/os-release
LTS_DRIVER_VERSION="2350.150"
wget -q https://repositories.intel.com/gpu/ubuntu/dists/${VERSION_CODENAME}/lts/${LTS_DRIVER_VERSION}/intel-gpu-ubuntu-${VERSION_CODENAME}-${LTS_DRIVER_VERSION}.run -O lts_driver.run
chmod +x lts_driver.run
./lts_driver.run -y
sudo ./lts_driver.run -r -y
sudo apt-get update
sudo apt install -yqq  \
    intel-opencl-icd intel-level-zero-gpu level-zero \
    intel-media-va-driver-non-free libmfxgen1 libvpl2 \
    libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
    libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
    mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo hwinfo clinfo
sudo rm -rf /var/lib/apt/lists/*
rm lts_driver.run
