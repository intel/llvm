// RUN: %{build} -o %t1.out
// RUN: %{run} %t1.out

//==-- pfor_flatten_range_shortcut.cpp - Kernel Launch Flattening test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

int main() {
  constexpr int n = 42;
  constexpr int magic_num = 4;

  sycl::queue q;
  auto ctxt = q.get_context();

  int *array = (int *)sycl::malloc_host(n * sizeof(int), q);

  if (array == nullptr) {
    return -1;
  }

  auto e1 = q.parallel_for(n, [=](auto item) { array[item] = 1; });

  auto e2 = q.parallel_for({n}, e1, [=](auto item) { array[item] += 2; });

  q.parallel_for(n, {e1, e2}, [=](auto item) { array[item]++; });

  q.wait();

  for (int i = 0; i < n; i++) {
    if (array[i] != magic_num) {
      assert(array[i] == magic_num);
    }
  }

  sycl::free(array, ctxt);

  return 0;
}
