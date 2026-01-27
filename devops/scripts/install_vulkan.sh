set -x
VULKAN_VER="1.4.335.0"
wget https://sdk.lunarg.com/sdk/download/$VULKAN_VER/linux/vulkansdk-linux-x86_64-$VULKAN_VER.tar.xz -O vulkan.tar.xz
tar xf vulkan.tar.xz
mv $VULKAN_VER vulkan
cd vulkan
sudo bash -c 'echo -e "APT::Get::Assume-Yes \"true\";\nAPT::Get::force-yes \"true\";" > /etc/apt/apt.conf.d/90forceyes'
sudo DEBIAN_FRONTEND=noninteractive ./vulkansdk --maxjobs
cd ..
rm vulkan.tar.xz
sudo rm /etc/apt/apt.conf.d/90forceyes
sudo mv vulkan /opt/
sudo bash -c 'echo "CMAKE_PREFIX_PATH=/opt/vulkan/x86_64/lib/cmake/" >> /etc/environment'
