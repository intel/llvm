//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// SYCL ordered queue event release shortcut test
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include <sycl/detail/core.hpp>

#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

int main() {
  queue q{property::queue::in_order()};
  auto dev = q.get_device();
  auto ctx = q.get_context();
  const int N = 8;

  if (dev.get_info<info::device::usm_shared_allocations>()) {
    auto A = (int *)malloc_shared(N * sizeof(int), dev, ctx);

    for (int i = 0; i < N; i++) {
      A[i] = 1;
    }

    {
      id<1> offset(0);
      q.parallel_for<class Baz>(range<1>{N}, offset, [=](id<1> ID) {
        auto i = ID[0];
        A[i]++;
      });

      q.wait();
    }

    {
      nd_range<1> NDR(range<1>{N}, range<1>{2});
      q.parallel_for<class NDFoo>(NDR, [=](nd_item<1> Item) {
        auto i = Item.get_global_id(0);
        A[i]++;
      });

      q.wait();
    }

    for (int i = 0; i < N; i++) {
      if (A[i] != 3)
        return 1;
    }
    free(A, ctx);
  }

  return 0;
}
