#!/usr/bin/env bash
# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#
# install_libbacktrace.sh - builds and installs tracing library
#

set -e

if [ "${SKIP_LIBBACKTRACE_BUILD}" ]; then
	echo "Variable 'SKIP_LIBBACKTRACE_BUILD' is set; skipping building libbacktrace"
	exit
fi

git clone https://github.com/ianlancetaylor/libbacktrace.git
pushd libbacktrace
./configure
make -j$(nproc)
make install -j$(nproc)
popd
