// UNSUPPORTED: cuda
// CUDA does not support the unnamed lambda extension.
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  -fsycl-unnamed-lambda %s -o %t1.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out

//==-- pfor_flatten_range_shortcut.cpp - Kernel Launch Flattening test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

int main() {
  int *array = nullptr;
  const int N = 42;
  const int MAGIC_NUM = 4;

  sycl::queue q;
  auto ctxt = q.get_context();

  array = (int *)sycl::malloc_host(N * sizeof(int), q);
  if (array == nullptr) {
    return -1;
  }

  auto e1 = q.parallel_for(N, [=](auto item) {
    array[item] = 1;
  });

  auto e2 = q.parallel_for({N}, e1, [=](auto item) {
    array[item] += 2;
  });

  q.parallel_for(N, {e1, e2}, [=](auto item) {
    array[item]++;
  });

  q.wait();

  for (int i = 0; i < N; i++) {
    if (array[i] != MAGIC_NUM) {
      assert(array[i] == MAGIC_NUM);
    }
  }

  free(array, ctxt);

  return 0;
}
