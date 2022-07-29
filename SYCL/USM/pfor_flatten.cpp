// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-unnamed-lambda -fsycl-dead-args-optimization %s -o %t1.out
// RUN: %HOST_RUN_PLACEHOLDER %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out

//==--------------- pfor_flatten.cpp - Kernel Launch Flattening test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

using namespace sycl;

class foo;
int main() {
  int *array = nullptr;
  const int N = 42;
  const int MAGIC_NUM = 42;

  queue q;
  auto ctxt = q.get_context();

  array = (int *)malloc_host(N * sizeof(int), q);
  if (array == nullptr) {
    return -1;
  }

  range<1> R{N};
  auto e1 = q.parallel_for(R, [=](id<1> ID) {
    int i = ID[0];
    array[i] = MAGIC_NUM - 4;
  });

  auto e2 = q.parallel_for(R, e1, [=](id<1> ID) {
    int i = ID[0];
    array[i] += 2;
  });

  auto e3 =
      q.parallel_for(nd_range<1>{R, range<1>{1}}, {e1, e2}, [=](nd_item<1> ID) {
        int i = ID.get_global_id(0);
        array[i]++;
      });

  auto e4 = q.single_task({e3}, [=]() {
    for (int i = 0; i < N; i++) {
      array[i]++;
    }
  });

  q.single_task(e4, [=]() { array[0] = array[0]; });

  q.wait();

  for (int i = 0; i < N; i++) {
    if (array[i] != MAGIC_NUM) {
      return -1;
    }
  }
  free(array, ctxt);

  return 0;
}
