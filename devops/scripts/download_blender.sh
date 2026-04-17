#!/bin/bash

set -x
set -e
set -o pipefail

apt update && apt install -yqq git-lfs

git clone -b v5.1.0 https://projects.blender.org/blender/blender.git
cd blender
make update
rm -rf .git
# Remove deps we will be building.
rm -rf lib/linux_x64/dpcpp
rm -rf lib/linux_x64/embree
rm -rf lib/linux_x64/level-zero
rm -rf lib/linux_x64/openimagedenoise
cd ..

tar -cJf blender_5_1_0.tar.xz blender
rm -rf blender
mv blender_5_1_0.tar.xz /opt/
