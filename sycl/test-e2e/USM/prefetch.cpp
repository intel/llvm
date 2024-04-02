//==---- prefetch.cpp - USM prefetch test ----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %{build} -o %t1.out
// RUN: %{run} %t1.out

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

static constexpr int count = 100;

int main() {
  queue q([](exception_list el) {
    for (auto &e : el)
      throw e;
  });
  if (q.get_device().get_info<info::device::usm_shared_allocations>()) {
    float *src = (float *)malloc_shared(sizeof(float) * count, q.get_device(),
                                        q.get_context());
    float *dest = (float *)malloc_shared(sizeof(float) * count, q.get_device(),
                                         q.get_context());
    for (int i = 0; i < count; i++)
      src[i] = i;

    // Test handler::prefetch
    {
      event init_prefetch = q.submit(
          [&](handler &cgh) { cgh.prefetch(src, sizeof(float) * count); });

      q.submit([&](handler &cgh) {
        cgh.depends_on(init_prefetch);
        cgh.single_task<class double_dest>([=]() {
          for (int i = 0; i < count; i++)
            dest[i] = 2 * src[i];
        });
      });
      q.wait_and_throw();

      for (int i = 0; i < count; i++) {
        assert(dest[i] == i * 2);
      }
    }

    // Test queue::prefetch
    {
      event init_prefetch = q.prefetch(src, sizeof(float) * count);

      q.submit([&](handler &cgh) {
        cgh.depends_on(init_prefetch);
        cgh.single_task<class double_dest3>([=]() {
          for (int i = 0; i < count; i++)
            dest[i] = 3 * src[i];
        });
      });
      q.wait_and_throw();

      for (int i = 0; i < count; i++) {
        assert(dest[i] == i * 3);
      }
    }
    free(src, q);
    free(dest, q);
  }
  return 0;
}
