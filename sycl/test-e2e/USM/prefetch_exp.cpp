//==-------- prefetch_exp.cpp - Experimental 2-way USM prefetch test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %{build} -o %t1.out
// RUN: %{run} %t1.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

static constexpr int Count = 100;

int main() {
  queue q([](exception_list el) {
    for (auto &e : el)
      throw e;
  });

  if (!q.get_device().get_info<info::device::usm_shared_allocations>()) {
    // USM not supported, skipping test and returning early.
    return 0;
  }

  float *Src = (float *)malloc_shared(sizeof(float) * Count, q.get_device(),
                                      q.get_context());
  float *Dest = (float *)malloc_shared(sizeof(float) * Count, q.get_device(),
                                        q.get_context());
  for (int i = 0; i < Count; i++)
    Src[i] = i;

  {
    // Test host-to-device prefetch via prefetch(handler ...).
    event InitPrefetch =
        ext::oneapi::experimental::submit_with_event(q, [&](handler &CGH) {
          ext::oneapi::experimental::prefetch(CGH, Src,
                                              sizeof(float) * Count);
        });

    q.submit([&](handler &CGH) {
      CGH.depends_on(init_prefetch);
      CGH.single_task<class double_dest>([=]() {
        for (int i = 0; i < Count; i++)
          Dest[i] = 2 * Src[i];
      });
    });
    q.wait_and_throw();

    for (int i = 0; i < Count; i++) {
      assert(Dest[i] == i * 2);
    }

    // Test device-to-host prefetch via prefetch(handler ...).
    q.submit([&](handler &CGH) {
      CGH.single_task<class quadruple_dest>([=]() {
        for (int i = 0; i < Count; i++)
          Dest[i] = 4 * Src[i];
      });
    });
    event InitPrefetchBack =
        ext::oneapi::experimental::submit_with_event(q, [&](handler &CGH) {
          ext::oneapi::experimental::prefetch(
              CGH, Src, sizeof(float) * Count,
              ext::oneapi::experimental::prefetch_type::host);
        });
    q.wait_and_throw();

    for (int i = 0; i < Count; i++) {
      assert(Dest[i] == i * 4);
    }
  }

  {
    // Test host-to-device prefetch via prefetch(queue ...).
    ext::oneapi::experimental::prefetch(
        q, Src, sizeof(float) * Count,
        ext::oneapi::experimental::prefetch_type::device);
    q.wait_and_throw();
    q.submit([&](handler &CGH) {
      CGH.single_task<class triple_dest>([=]() {
        for (int i = 0; i < Count; i++)
          Dest[i] = 3 * Src[i];
      });
    });
    q.wait_and_throw();

    for (int i = 0; i < Count; i++) {
      assert(Dest[i] == i * 3);
    }

    // Test device-to-host prefetch via prefetch(queue ...).
    q.submit([&](handler &CGH) {
      CGH.single_task<class sixtuple_dest>([=]() {
        for (int i = 0; i < Count; i++)
          Dest[i] = 6 * Src[i];
      });
    });
    q.wait_and_throw();
    ext::oneapi::experimental::prefetch(
        q, Src, sizeof(float) * Count,
        ext::oneapi::experimental::prefetch_type::host);
    q.wait_and_throw();

    for (int i = 0; i < Count; i++) {
      assert(Dest[i] == i * 6);
    }
  }
  free(Src, q);
  free(Dest, q);
}
