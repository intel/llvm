#!/bin/bash

set -x
set -e
set -o pipefail
apt update && apt install -yqq qt6-wayland-dev-tools libglm-dev cmake libxcb-dri3-0 libxcb-present0 libpciaccess0 \
libpng-dev libxcb-keysyms1-dev libxcb-dri3-dev libx11-dev g++ gcc \
libwayland-dev libxrandr-dev libxcb-randr0-dev libxcb-ewmh-dev \
git python-is-python3 bison libx11-xcb-dev liblz4-dev libzstd-dev \
ocaml-core ninja-build pkg-config libxml2-dev wayland-protocols python3-jsonschema \
clang-format qtbase5-dev qt6-base-dev qt6-wayland-dev
VULKAN_VER="1.4.357.1"
wget https://sdk.lunarg.com/sdk/download/$VULKAN_VER/linux/vulkansdk-linux-x86_64-$VULKAN_VER.tar.xz -O vulkan.tar.xz
tar xf vulkan.tar.xz
mv $VULKAN_VER vulkan
cd vulkan
sudo bash -c 'echo -e "APT::Get::Assume-Yes \"true\";\nAPT::Get::force-yes \"true\";" > /etc/apt/apt.conf.d/90forceyes'
sudo DEBIAN_FRONTEND=noninteractive ./vulkansdk --maxjobs
# Delete huge directory of unneeded build artifacts
sudo rm -r source
cd ..
rm vulkan.tar.xz
sudo rm /etc/apt/apt.conf.d/90forceyes
sudo mv vulkan /opt/
sudo bash -c 'echo "CMAKE_PREFIX_PATH=/opt/vulkan/x86_64/lib/cmake/" >> /etc/environment'
