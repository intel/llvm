#!/bin/bash

set -x
set -e
set -o pipefail
VULKAN_VER="1.4.335.0"
VULKAN_SDK_INSTALL_DIR="${VULKAN_SDK_INSTALL_DIR:-/opt/vulkan}"
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
sudo mv vulkan "${VULKAN_SDK_INSTALL_DIR}"
VULKAN_SDK_PATH="${VULKAN_SDK_INSTALL_DIR}/x86_64"
if [ ! -d "${VULKAN_SDK_PATH}" ]; then
  VULKAN_SDK_PATH="${VULKAN_SDK_INSTALL_DIR}"
fi
VALIDATION_LAYER_MANIFEST="$(find "${VULKAN_SDK_PATH}" -path "*/explicit_layer.d/VkLayer_khronos_validation.json" -print -quit)"
sudo bash -c "echo \"CMAKE_PREFIX_PATH=${VULKAN_SDK_PATH}/lib/cmake/\" >> /etc/environment"
sudo mkdir -p /etc/vulkan/explicit_layer.d
if [ -n "${VALIDATION_LAYER_MANIFEST}" ] && [ -f "${VALIDATION_LAYER_MANIFEST}" ]; then
  sudo ln -sf "${VALIDATION_LAYER_MANIFEST}" \
    /etc/vulkan/explicit_layer.d/VkLayer_khronos_validation.json
else
  echo "Warning: Vulkan validation layer manifest not found under ${VULKAN_SDK_PATH}. Validation layers will not be available; verify the ${VULKAN_VER} SDK download and layout." >&2
fi
echo "${VULKAN_SDK_PATH}/lib" | sudo tee /etc/ld.so.conf.d/vulkan-sdk.conf >/dev/null
sudo ldconfig
