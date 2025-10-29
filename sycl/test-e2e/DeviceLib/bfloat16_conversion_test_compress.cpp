//==-------------- bf1oat16 devicelib test for SYCL JIT --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check bfloat16 devicelib device image compression.

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20397

// REQUIRES: linux, zstd
// RUN: rm -rf %t.dir; mkdir -p %t.dir
// RUN: %{build} --offload-compress -DBUILD_LIB -fPIC -shared -o %t.dir/lib%basename_t_compress.so
// RUN: %{build} --offload-compress -DBUILD_EXE -L%t.dir -o %t1.out -l%basename_t_compress -Wl,-rpath=%t.dir
// RUN: %{run} %t1.out

// UNSUPPORTED: target-nvidia || target-amd
// UNSUPPORTED-INTENDED: bfloat16 device library is not used on AMD and Nvidia.

#include "bfloat16_conversion_test.hpp"
