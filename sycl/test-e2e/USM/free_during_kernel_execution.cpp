//==------------------- free_during_kernel_execution.cpp -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RUN: %{build} -o %t1.out
// RUN: %{run} %t1.out

// Windows doesn't yet have full shutdown().
// UNSUPPORTED: ze_debug && windows

#include <sycl/sycl.hpp>

class KernelA;

int main() {
  const int N = 512;
  sycl::queue Queue;
  sycl::buffer<int, 1> Buffer(N);
  sycl::range<1> NumOfWorkItems{Buffer.size()};

  auto *USM = sycl::malloc_host<int>(1, Queue.get_context());

  Queue.submit([&](sycl::handler &cgh) {
    auto Accessor = Buffer.get_access<sycl::access::mode::write>(cgh);
    cgh.single_task<KernelA>([=]() {
      for (int I = 0; I < N; ++I) {
        Accessor[I] = I;
      }
    });
  });

  // Check that freeing USM before the kernel was finished works.
  free(USM, Queue.get_context());

  return 0;
}
