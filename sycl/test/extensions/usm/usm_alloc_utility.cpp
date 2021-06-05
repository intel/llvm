// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: %RUN_ON_HOST %t1.out

//==------ usm_alloc_utility.cpp - USM malloc and aligned_alloc test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <cassert>

using namespace cl::sycl;

constexpr int N = 8;

int main() {
  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();
  int *array;

  if (dev.get_info<info::device::usm_host_allocations>()) {
    array = (int *)malloc_host(N * sizeof(int), q);
    assert((get_pointer_type(array, ctxt) == usm::alloc::host) &&
           "Allocation pointer should be host type");
    assert((get_pointer_device(array, ctxt) == dev) &&
           "Allocation pointer should be host type");
    free(array, ctxt);

    array =
        (int *)aligned_alloc_host(alignof(long long), N * sizeof(int), ctxt);
    assert((get_pointer_type(array, ctxt) == usm::alloc::host) &&
           "Allocation pointer should be host type");
    assert((get_pointer_device(array, ctxt) == dev) &&
           "Allocation pointer should be host type");
    free(array, ctxt);
  }

  if (dev.get_info<info::device::usm_shared_allocations>()) {
    array = (int *)malloc_shared(N * sizeof(int), q);
    // host device treats all allocations as host allocations
    assert((get_pointer_type(array, ctxt) == usm::alloc::host) &&
           "Allocation pointer should be host type");
    assert((get_pointer_device(array, ctxt) == dev) &&
           "Allocation pointer should be host type");
    free(array, ctxt);

    array = (int *)aligned_alloc_shared(alignof(long long), N * sizeof(int),
                                        dev, ctxt);
    // host device treats all allocations as host allocations
    assert((get_pointer_type(array, ctxt) == usm::alloc::host) &&
           "Allocation pointer should be host type");
    assert((get_pointer_device(array, ctxt) == dev) &&
           "Allocation pointer should be host type");
    free(array, ctxt);
  }

  if (dev.get_info<info::device::usm_device_allocations>()) {
    array = (int *)malloc_device(N * sizeof(int), q);
    // host device treats all allocations as host allocations
    assert((get_pointer_type(array, ctxt) == usm::alloc::host) &&
           "Allocation pointer should be host type");
    assert((get_pointer_device(array, ctxt) == dev) &&
           "Allocation pointer should be host type");
    free(array, ctxt);

    array = (int *)aligned_alloc_device(alignof(long long), N * sizeof(int),
                                        dev, ctxt);
    // host device treats all allocations as host allocations
    assert((get_pointer_type(array, ctxt) == usm::alloc::host) &&
           "Allocation pointer should be host type");
    assert((get_pointer_device(array, ctxt) == dev) &&
           "Allocation pointer should be host type");
    free(array, ctxt);
  }

  return 0;
}
