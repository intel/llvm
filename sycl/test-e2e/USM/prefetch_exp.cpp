//==---- prefetch_exp.cpp - Experimental 2-way USM prefetch test
//------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %{build} -o %t1.out
// RUN: %{run} %t1.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/USM/prefetch_exp.hpp>
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

    {
      // Test host to device handler::ext_oneapi_prefetch_exp
      event init_prefetch = q.submit([&](handler &cgh) {
        cgh.ext_oneapi_prefetch_exp(src, sizeof(float) * count);
      });

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

      // Test device to host handler::ext_oneapi_prefetch_exp
      q.submit([&](handler &cgh) {
        cgh.single_task<class quadruple_dest>([=]() {
          for (int i = 0; i < count; i++)
            dest[i] = 4 * src[i];
        });
      });
      event init_prefetch_back = q.submit([&](handler &cgh) {
        cgh.ext_oneapi_prefetch_exp(dest, sizeof(float) * count,
                                    sycl::ext::oneapi::experimental::
                                        migration_direction::DEVICE_TO_HOST);
      });
      q.wait_and_throw();

      for (int i = 0; i < count; i++) {
        assert(dest[i] == i * 4);
      }
    }

    // Test queue::prefetch
    {
      event init_prefetch =
          q.ext_oneapi_prefetch_exp(src, sizeof(float) * count);

      q.submit([&](handler &cgh) {
        cgh.depends_on(init_prefetch);
        cgh.single_task<class triple_dest>([=]() {
          for (int i = 0; i < count; i++)
            dest[i] = 3 * src[i];
        });
      });
      q.wait_and_throw();

      for (int i = 0; i < count; i++) {
        assert(dest[i] == i * 3);
      }

      q.submit([&](handler &cgh) {
        cgh.single_task<class sixtuple_dest>([=]() {
          for (int i = 0; i < count; i++)
            dest[i] = 6 * src[i];
        });
      });
      event init_prefetch_back = q.ext_oneapi_prefetch_exp(
          src, sizeof(float) * count,
          sycl::ext::oneapi::experimental::migration_direction::DEVICE_TO_HOST);
      q.wait_and_throw();

      for (int i = 0; i < count; i++) {
        assert(dest[i] == i * 6);
      }
    }
    free(src, q);
    free(dest, q);
  }
  return 0;
}
