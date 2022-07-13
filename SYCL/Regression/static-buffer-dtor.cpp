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
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// Failing on HIP AMD
// XFAIL: hip_amd

#include <sycl/sycl.hpp>

int main() {
  uint8_t *h_A = (uint8_t *)malloc(256);
  static cl::sycl::buffer<uint8_t> bufs[2] = {cl::sycl::range<1>(256),
                                              cl::sycl::range<1>(256)};
  cl::sycl::queue q;
  q.submit([&](cl::sycl::handler &cgh) {
    cgh.copy(h_A, bufs[0].get_access<cl::sycl::access::mode::write>(cgh));
  });
  q.submit([&](cl::sycl::handler &cgh) {
    cgh.copy(h_A, bufs[1].get_access<cl::sycl::access::mode::write>(cgh));
  });
  return 0;
}
