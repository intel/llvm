//==- device_is_integrated_gpu.cpp - sycl_ext_oneapi_device_is_integrated_gpu
// test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-gen12
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test checks that aspect::ext_oneapi_is_integrated_gpu is true if GPU device
// is integrated.

#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {
  queue Queue;
  auto dev = Queue.get_device();

  if (dev.has(aspect::ext_oneapi_is_integrated_gpu))
    return 0;

  assert(false && "aspect::ext_oneapi_is_integrated_gpu must be true");
  return 1;
}
