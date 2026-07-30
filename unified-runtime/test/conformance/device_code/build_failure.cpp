// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
