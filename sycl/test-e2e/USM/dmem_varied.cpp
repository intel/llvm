// RUN: %{build} -o %t1.out
// RUN: %{run} %t1.out

// This test is expected to reliably work with USM allocator which is
// currently enabled only on level zero.
// REQUIRES: level_zero

//==---------- dmem_varied.cpp - Test various sizes and alignments ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <vector>

using namespace sycl;

int main() {
  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();

  if (!dev.get_info<info::device::usm_device_allocations>() ||
      !dev.get_info<info::device::usm_shared_allocations>()) {
    return 0;
  }

  // Check allocation on small sizes and a large one, For each allocation
  // also check several different alignments.
  // To verify on device, store a valu per each pointer and calculate
  // the sum on device and then check it.

  constexpr size_t smallSizeLimit = 256;
  constexpr size_t alignmentLimit = 64;
  constexpr size_t largeSize = 128 * 1024; // 128k

  // 1000 should be enough to store all the allocated pointers
  constexpr size_t numPtrs = 1000;

  // Allocate as shared, this memory is used to store device pointers on host
  // and pass them to device
  uint8_t **ptrs = (uint8_t **)malloc_shared(numPtrs * sizeof(uint8_t *), q);
  assert(ptrs != nullptr);
  memset(ptrs, 0, numPtrs * sizeof(uint8_t *));

  size_t count = 0;

  // Small sizes to allocate
  // Allocated sizes 2^n - 1 up to smallSizeLimit
  for (size_t size = 2; size <= smallSizeLimit; size *= 2) {
    uint8_t *p = (uint8_t *)malloc_device(size - 1, q);
    assert(p != nullptr);
    ptrs[count++] = p;

    // Also test cases with alignment > size
    for (size_t alignment = 1; alignment <= alignmentLimit; alignment *= 2) {
      uint8_t *s = (uint8_t *)aligned_alloc_device(alignment, size - 1, q);
      assert(s != nullptr);
      assert(((size_t)s) % alignment == 0);

      ptrs[count++] = s;
    }
  }

  ptrs[count] = (uint8_t *)malloc_device(largeSize, q);
  assert(ptrs[count]);
  count++;

  for (size_t alignment = 1; alignment <= alignmentLimit; alignment *= 8) {
    uint8_t *a = (uint8_t *)aligned_alloc_device(alignment, largeSize, q);
    assert(a);
    assert(((size_t)a) % alignment == 0);

    ptrs[count++] = a;
  }

  q.submit([&](handler &h) {
     h.single_task<class foo1>([=]() {
       for (size_t i = 0; i < count; ++i) {
         *ptrs[i] = 1;
       }
     });
   }).wait();

  size_t *res =
      (size_t *)aligned_alloc_shared(alignof(size_t), sizeof(size_t), q);
  assert(res);
  assert(((size_t)res) % alignof(size_t) == 0);
  *res = 0;

  q.submit([&](handler &h) {
     h.single_task<class foo>([=]() {
       for (size_t i = 0; i < count; ++i) {
         *res += *ptrs[i];
       }
     });
   }).wait();

  assert(*res == count);

  for (size_t i = 0; i < numPtrs; ++i) {
    if (ptrs[i] != nullptr) {
      free(ptrs[i], q);
    }
  }

  free(res, q);
  free(ptrs, q);

  return 0;
}
