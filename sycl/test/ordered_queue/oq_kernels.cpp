// RUN: %clangxx -fsycl -fsycl-unnamed-lambda %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//==------ oq_kernels.cpp - SYCL ordered queue kernel shortcut test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <iostream>
using namespace cl::sycl;

int main() {
  ordered_queue q;
  auto dev = q.get_device();
  auto ctx = q.get_context();
  const int N = 8;

  auto A = (int *)malloc_shared(N * sizeof(int), dev, ctx);

  for (int i = 0; i < N; i++) {
    A[i] = 1;
  }

  q.parallel_for(range<1>{N}, [=](id<1> ID) {
    auto i = ID[0];
    A[i]++;
  });

  q.parallel_for<class Foo>(range<1>{N}, [=](id<1> ID) {
    auto i = ID[0];
    A[i]++;
  });

  q.single_task<class Bar>([=]() {
    for (int i = 0; i < N; i++) {
      A[i]++;
    }
  });

  id<1> offset(0);
  q.parallel_for<class Baz>(range<1>{N}, offset, [=](id<1> ID) {
    auto i = ID[0];
    A[i]++;
  });

  nd_range<1> NDR(range<1>{N}, range<1>{2});
  q.parallel_for<class NDFoo>(NDR, [=](id<1> ID) {
    auto i = ID[0];
    A[i]++;
  });

  q.wait();

  for (int i = 0; i < N; i++) {
    if (A[i] != 6)
      return 1;
  }

  return 0;
}
