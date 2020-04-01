// RUN: %clangxx -fsycl %s -o %t1.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t1.out

//==-------------- pointer_query.cpp - Pointer Query test ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {
  int *array = nullptr;
  const int N = 4;
  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();

  if (!(dev.get_info<info::device::usm_device_allocations>() &&
        dev.get_info<info::device::usm_shared_allocations>() &&
        dev.get_info<info::device::usm_host_allocations>()))
    return 0;

  usm::alloc Kind;
  device D;

  // Test device allocs
  array = (int *)malloc_device(N * sizeof(int), q);
  if (array == nullptr) {
    return 1;
  }
  Kind = get_pointer_type(array, ctxt);
  if (ctxt.is_host()) {
    // for now, host device treats all allocations
    // as host allocations
    if (Kind != usm::alloc::host) {
      return 2;
    }
  } else {
    if (Kind != usm::alloc::device) {
      return 3;
    }
  }
  D = get_pointer_device(array, ctxt);
  if (D != dev) {
    return 4;
  }
  free(array, ctxt);

  // Test shared allocs
  array = (int *)malloc_shared(N * sizeof(int), q);
  if (array == nullptr) {
    return 5;
  }
  Kind = get_pointer_type(array, ctxt);
  if (ctxt.is_host()) {
    // for now, host device treats all allocations
    // as host allocations
    if (Kind != usm::alloc::host) {
      return 6;
    }
  } else {
    if (Kind != usm::alloc::shared) {
      return 7;
    }
  }
  D = get_pointer_device(array, ctxt);
  if (D != dev) {
    return 8;
  }
  free(array, ctxt);

  // Test host allocs
  array = (int *)malloc_host(N * sizeof(int), q);
  if (array == nullptr) {
    return 9;
  }
  Kind = get_pointer_type(array, ctxt);
  if (Kind != usm::alloc::host) {
    return 10;
  }
  D = get_pointer_device(array, ctxt);
  auto Devs = ctxt.get_devices();
  auto result = std::find(Devs.begin(), Devs.end(), D);
  if (result == Devs.end()) {
    // Returned device was not in queried context
    return 11;
  }
  free(array, ctxt);

  // Test invalid ptrs
  Kind = get_pointer_type(nullptr, ctxt);
  if (Kind != usm::alloc::unknown) {
    return 11;
  }

  // next checks only valid for non-host contexts
  array = (int*)malloc(N*sizeof(int));
  Kind = get_pointer_type(array, ctxt);
  if (!ctxt.is_host()) {
    if (Kind != usm::alloc::unknown) {
      return 12;
    }
    try {
      D = get_pointer_device(array, ctxt);
    } catch (runtime_error) {
      return 0;
    }
    return 13;
  } else {
    // host ctxts always report host
    if (Kind != usm::alloc::host) {
      return 14;
    }
  }
  free(array);

  return 0;
}
