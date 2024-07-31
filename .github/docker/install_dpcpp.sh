#!/usr/bin/env bash
# Copyright (C) 2023-2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#
# install_dpcpp.sh - unpacks DPC++ in ${DPCPP_PATH}, to be used while building UR
#

set -e

if [ "${SKIP_DPCPP_BUILD}" ]; then
	echo "Variable 'SKIP_DPCPP_BUILD' is set; skipping building DPC++"
	exit
fi

mkdir -p ${DPCPP_PATH}/dpcpp_compiler
wget -O ${DPCPP_PATH}/dpcpp_compiler.tar.gz https://github.com/intel/llvm/releases/download/nightly-2024-01-29/sycl_linux.tar.gz
tar -xvf ${DPCPP_PATH}/dpcpp_compiler.tar.gz -C ${DPCPP_PATH}/dpcpp_compiler
