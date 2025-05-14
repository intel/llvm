//==-- static-buffer-dtor.cpp ----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This checks that the runtime does not crash if there are SYCL buffer
// destructors that run as part of program shutdown, after the runtime itself
// would start shutting down.
//===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Failing on HIP AMD
// UNSUPPORTED: hip

// Windows doesn't yet have full shutdown().
// UNSUPPORTED: ze_debug && windows

#include <sycl/detail/core.hpp>

int main() {

  static sycl::buffer<uint8_t> bufs[2] = {sycl::range<1>(256),
                                          sycl::range<1>(256)};
  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
    auto acc = bufs[0].get_access<sycl::access::mode::write>(cgh);
    cgh.single_task([=]() {
      for (int i = 0; i < 256; i++) {
        acc[i] = 24;
      }
    });
  });

  q.submit([&](sycl::handler &cgh) {
    auto acc = bufs[1].get_access<sycl::access::mode::write>(cgh);
    cgh.single_task([=]() {
      for (int i = 0; i < 256; i++) {
        acc[i] = 25;
      }
    });
  });

  // no q.wait()
  return 0;
}
