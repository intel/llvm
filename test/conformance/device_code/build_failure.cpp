// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include <sycl/sycl.hpp>

int main() {
  sycl::queue deviceQueue;

  auto Kernel = []() {
#ifdef __SYCL_DEVICE_ONLY__
    asm volatile("undefined\n");
#endif // __SYCL_DEVICE_ONLY__
  };

  deviceQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class Foo>(Kernel); });

  return 0;
}
