//==-------------- bf1oat16 devicelib test for SYCL JIT --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: linux
// RUN: %{build} -DBUILD_LIB -fPIC -shared -o %T/lib%basename_t.so
// RUN: %{build} -DBUILD_EXE -L%T -o %t1.out -l%basename_t -Wl,-rpath=%T
// RUN: %{run} %t1.out

// UNSUPPORTED: target-nvidia || target-amd
// UNSUPPORTED-INTENDED: bfloat16 device library is not used on AMD and Nvidia.

#include "bfloat16_conversion_test.hpp"
