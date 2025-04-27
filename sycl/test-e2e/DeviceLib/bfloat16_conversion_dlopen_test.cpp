//==----------- bf1oat16 devicelib dlopen test for SYCL JIT ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The case uses dlopen/close to load/unload a sycl shared library which
// depends bfloat16 device library and the main function also includes sycl
// kernels which depend on bfloat16 device library. SYCL program manager will
// own the bfloat16 device library image which is shared by all kernels using
// bfloat16 features, so the program should also work well when the shared
// library is dlclosed and the device images are removed.

// REQUIRES: linux

// RUN: %{build} -DBUILD_LIB -fPIC -shared -o %T/lib%basename_t.so

// RUN: %{build} -DFNAME=%basename_t -ldl -Wl,-rpath=%T -o %t1.out

// RUN: %{run} %t1.out

// UNSUPPORTED: target-nvidia || target-amd
// UNSUPPORTED-INTENDED: bfloat16 device library is not used on AMD and Nvidia.

#include "bfloat16_conversion_dlopen_test.hpp"
