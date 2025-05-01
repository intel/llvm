//==---------- bf1oat16 devicelib dlopen_test_compress for SYCL JIT --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check bfloat16 devicelib device image compression.

// REQUIRES: linux, zstd
// RUN: %{build} --offload-compress -DBUILD_LIB -fPIC -shared -o %T/lib%basename_t_compress.so
// RUN: %{build} --offload-compress -DFNAME=%basename_t_compress -ldl -o %t1.out -Wl,-rpath=%T
// RUN: %{run} %t1.out

// UNSUPPORTED: target-nvidia || target-amd
// UNSUPPORTED-INTENDED: bfloat16 device library is not used on AMD and Nvidia.

#include "bfloat16_conversion_dlopen_test.hpp"
