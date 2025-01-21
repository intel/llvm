// RUN: %{build} -o %t1.out
// RUN: %{run} %t1.out

// This test is expected to reliably work with USM allocator which is
// currently enabled only on level zero.
// REQUIRES: level_zero

//==------ smem_concurrent.cpp - Concurrent USM allocation test -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <array>
#include <vector>

using namespace sycl;

const int N = 8;

class foo;
int main() {
  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();

  if (!dev.get_info<info::device::usm_shared_allocations>()) {
    return 1;
  }

  // This test checks that we're able to allocate/deallocate shared memory while
  // the kernel is running, but we don't actually access such memory on the
  // host.
  constexpr size_t allocSize = 100;
  constexpr size_t numAllocs = 6;

  // Allocate "host" and "device" arrays of pointers.
  uint8_t **hostPtrs =
      (uint8_t **)malloc_shared(numAllocs * sizeof(uint8_t *), q);
  uint8_t **devicePtrs =
      (uint8_t **)malloc_shared(numAllocs * sizeof(uint8_t *), q);

  // Fill "device" array with pointers to memory allocated with malloc_shared
  for (size_t idx = 0; idx < numAllocs; ++idx) {
    uint8_t *p = (uint8_t *)malloc_shared(allocSize, q);
    *p = 1;
    devicePtrs[idx] = p;
  }

  // Fill first halft of "host" array with pointers to memory allocated with
  // malloc_shared. This part will be freed later.
  for (size_t idx = 0; idx < numAllocs / 2; ++idx) {
    uint8_t *p = (uint8_t *)malloc_shared(allocSize, q);
    *p = 1;
    hostPtrs[idx] = p;
  }

  // Allocate a memory to store the result of computation.
  uint8_t *res = (uint8_t *)malloc_shared(1, q);
  *res = 0;

  // Run computation on device using "device" array
  auto e = q.submit([&](handler &h) {
    h.single_task<class foo>([res, devicePtrs]() {
      for (size_t i = 0; i < numAllocs; ++i) {
        *res += *(uint8_t *)devicePtrs[i];
      }
    });
  });

  // While running the computation kernel,
  // free first half of "host" array
  for (size_t i = 0; i < numAllocs / 2; ++i) {
    free(hostPtrs[i], q);
    hostPtrs[i] = nullptr;
  }

  // And then fill the second part of array with pointers allocated via
  // malloc_shared.
  for (size_t i = numAllocs / 2; i < numAllocs; ++i) {
    uint8_t *p = (uint8_t *)malloc_shared(allocSize, q);
    *p = 1;
    hostPtrs[i] = p;
  }

  e.wait();

  // After the kernel is finished update the computation result
  // with data from "host" array of ptrs.
  for (size_t i = 0; i < numAllocs; ++i) {
    if (hostPtrs[i] == nullptr) {
      *res += 2;
    } else {
      *res += *(uint8_t *)hostPtrs[i];
    }
  }

  // Check the result
  // +1 for each element in "device" array
  // +2 for each freed "host" array ptr
  // +1 for each allocated "host" array ptr
  //
  // total = 1 * numAllocs + numAllocs / 2 * 2 + numAllocs / 2
  assert(*res == (numAllocs * 2 + numAllocs / 2));

  for (size_t i = 0; i < numAllocs; ++i) {
    if (devicePtrs[i] != nullptr) {
      free(devicePtrs[i], q);
    }
    if (hostPtrs[i] != nullptr) {
      free(hostPtrs[i], q);
    }
  }

  free(res, q);
  free(devicePtrs, q);
  free(hostPtrs, q);

  return 0;
}
