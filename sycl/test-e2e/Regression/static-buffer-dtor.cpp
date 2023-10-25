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
// UNSUPPORTED: hip_amd

// Windows doesn't yet have full shutdown().
// UNSUPPORTED: ze_debug && windows

#include <sycl/sycl.hpp>

int main() {
  uint8_t *h_A = (uint8_t *)malloc(256);
  static sycl::buffer<uint8_t> bufs[2] = {sycl::range<1>(256),
                                          sycl::range<1>(256)};
  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
    cgh.copy(h_A, bufs[0].get_access<sycl::access::mode::write>(cgh));
  });
  q.submit([&](sycl::handler &cgh) {
    cgh.copy(h_A, bufs[1].get_access<sycl::access::mode::write>(cgh));
  });
  free(h_A);
  return 0;
}
