//==---- usm_import_release.cpp - USM L0 import/release test ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// REQUIRES: level_zero
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

static constexpr int N = 100;
static constexpr int VAL = 42;
static constexpr size_t SIZE = N * sizeof(int);

void init_on_host(int *arr) {
  for (int i = 0; i < N; ++i) {
    arr[i] = VAL + i;
  }
}

void check_on_host(int *arr) {
  for (int i = 0; i < N; ++i) {
    assert(arr[i] == VAL + i);
  }
}

#define MALLOC(ARR) ARR = (int *)malloc(SIZE)

#define USM_MALLOC(ARR, ALLOC_TYPE) ARR = (int *)malloc_##ALLOC_TYPE(SIZE, q)

void test_memcpy(queue &q, int *from, void from_init(int *), int *temp, int *to,
                 void check(int *)) {
  from_init(from);

  prepare_for_device_copy(from, SIZE, q);
  q.submit([&](handler &h) { h.memcpy(temp, from, SIZE); });
  q.wait();
  prepare_for_device_copy(to, SIZE, q);
  q.submit([&](handler &h) { h.memcpy(to, temp, SIZE); });
  q.wait();
  release_from_device_copy(from, q);
  release_from_device_copy(to, q);
  check(to);

  auto ctxt = q.get_context();
  prepare_for_device_copy(from, SIZE, ctxt);
  q.submit([&](handler &h) { h.memcpy(temp, from, SIZE); });
  q.wait();
  prepare_for_device_copy(to, SIZE, ctxt);
  q.submit([&](handler &h) { h.memcpy(to, temp, SIZE); });
  q.wait();
  release_from_device_copy(from, ctxt);
  release_from_device_copy(to, ctxt);
  check(to);

  free(from);
  free(temp, ctxt);
  free(to);
}

int main() {
  queue q;
  int *src;
  int *temp;
  int *dst;
  auto dev = q.get_device();

  if (dev.get_info<sycl::info::device::usm_host_allocations>()) {
    // Test system memory to/from USM host memory
    MALLOC(src);
    USM_MALLOC(temp, host);
    MALLOC(dst);
    test_memcpy(q, src, init_on_host, temp, dst, check_on_host);
  }

  if (dev.get_info<sycl::info::device::usm_shared_allocations>()) {
    // Test system memory to/from USM shared memory
    MALLOC(src);
    USM_MALLOC(temp, shared);
    MALLOC(dst);
    test_memcpy(q, src, init_on_host, temp, dst, check_on_host);
  }

  if (dev.get_info<sycl::info::device::usm_device_allocations>()) {
    // Test system memory to/from USM device memory
    MALLOC(src);
    USM_MALLOC(temp, device);
    MALLOC(dst);
    test_memcpy(q, src, init_on_host, temp, dst, check_on_host);
  }

  return 0;
}
