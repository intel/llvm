#!/bin/bash

set -x
set -e
set -o pipefail
VULKAN_VER="1.4.335.0"
VULKAN_SDK_ROOT="/opt/vulkan"
VULKAN_SDK_PATH="${VULKAN_SDK_ROOT}/x86_64"
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
sudo mv vulkan "${VULKAN_SDK_ROOT}"
sudo bash -c "echo \"CMAKE_PREFIX_PATH=${VULKAN_SDK_PATH}/lib/cmake/\" >> /etc/environment"
sudo mkdir -p /etc/vulkan/explicit_layer.d
if [ -f "${VULKAN_SDK_PATH}/etc/vulkan/explicit_layer.d/VkLayer_khronos_validation.json" ]; then
  sudo ln -sf "${VULKAN_SDK_PATH}/etc/vulkan/explicit_layer.d/VkLayer_khronos_validation.json" \
    /etc/vulkan/explicit_layer.d/VkLayer_khronos_validation.json
else
  echo "Warning: Vulkan validation layer manifest not found in SDK install; validation layers will not be available. Verify the SDK install completed and includes explicit_layer.d." >&2
fi
echo "${VULKAN_SDK_PATH}/lib" | sudo tee /etc/ld.so.conf.d/vulkan-sdk.conf >/dev/null
sudo ldconfig
