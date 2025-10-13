//==-------------- bf1oat16 devicelib test for SYCL JIT --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: linux
// RUN: rm -rf %t.dir; mkdir -p %t.dir
// RUN: %{build} -DBUILD_LIB -fPIC -shared -o %t.dir/lib%basename_t.so
// RUN: %{build} -DBUILD_EXE -L%t.dir -o %t1.out -l%basename_t -Wl,-rpath=%t.dir
// RUN: %{run} %t1.out

// UNSUPPORTED: target-nvidia || target-amd
// UNSUPPORTED-INTENDED: bfloat16 device library is not used on AMD and Nvidia.
// UNSUPPORTED: target-native_cpu
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20142

#include "bfloat16_conversion_test.hpp"
