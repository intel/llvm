#!/usr/bin/env bash

#  Copyright (C) 2024 Intel Corporation
#  Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
#  See LICENSE.TXT
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# install_hwloc.sh - Script for building and installing HWLOC library from source code

set -e

git clone -b hwloc-2.3.0 https://github.com/open-mpi/hwloc.git
pushd hwloc
./autogen.sh
./configure
make -j$(nproc)
sudo make install -j$(nproc)
popd
