// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==------ usm_alloc_utility.cpp - USM malloc and aligned_alloc test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <cassert>

using namespace sycl;

constexpr int N = 8;

static void check_and_free(int *array, const device &dev, const context &ctxt,
                           usm::alloc expected_type) {
  // host device treats all allocations as host allocations
  assert((get_pointer_type(array, ctxt) == expected_type) &&
         "Allocation pointer has unexpected type.");
  assert((get_pointer_device(array, ctxt) == dev) &&
         "Allocation pointer has unexpected device associated with it.");
  free(array, ctxt);
}

int main() {
  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();
  int *array;

  if (dev.get_info<info::device::usm_host_allocations>()) {
    array = (int *)malloc(N * sizeof(int), q, usm::alloc::host);
    check_and_free(array, dev, ctxt, usm::alloc::host);

    array =
        (int *)malloc(N * sizeof(int), q, usm::alloc::host, property_list{});
    check_and_free(array, dev, ctxt, usm::alloc::host);

    array = (int *)aligned_alloc(alignof(long long), N * sizeof(int), q,
                                 usm::alloc::host);
    check_and_free(array, dev, ctxt, usm::alloc::host);

    array = (int *)aligned_alloc(alignof(long long), N * sizeof(int), q,
                                 usm::alloc::host, property_list{});
    check_and_free(array, dev, ctxt, usm::alloc::host);

    array = (int *)malloc_host(N * sizeof(int), q);
    check_and_free(array, dev, ctxt, usm::alloc::host);

    array = (int *)malloc_host(
        N * sizeof(int), q,
        property_list{
            ext::intel::experimental::property::usm::buffer_location{2}});
    check_and_free(array, dev, ctxt, usm::alloc::host);

    array =
        (int *)aligned_alloc_host(alignof(long long), N * sizeof(int), ctxt);
    check_and_free(array, dev, ctxt, usm::alloc::host);

    array = (int *)aligned_alloc_host(
        alignof(long long), N * sizeof(int), ctxt,
        property_list{
            ext::intel::experimental::property::usm::buffer_location{2}});
    check_and_free(array, dev, ctxt, usm::alloc::host);
  }

  if (dev.get_info<info::device::usm_shared_allocations>()) {
    array = (int *)malloc_shared(N * sizeof(int), q);
    check_and_free(array, dev, ctxt, usm::alloc::shared);

    array = (int *)malloc_shared(
        N * sizeof(int), q,
        property_list{
            ext::intel::experimental::property::usm::buffer_location{2}});
    check_and_free(array, dev, ctxt, usm::alloc::shared);

    array = (int *)aligned_alloc_shared(alignof(long long), N * sizeof(int),
                                        dev, ctxt);
    check_and_free(array, dev, ctxt, usm::alloc::shared);

    array = (int *)aligned_alloc_shared(
        alignof(long long), N * sizeof(int), dev, ctxt,
        property_list{
            ext::intel::experimental::property::usm::buffer_location{2}});
    check_and_free(array, dev, ctxt, usm::alloc::shared);
  }

  if (dev.get_info<info::device::usm_device_allocations>()) {
    array = (int *)malloc_device(N * sizeof(int), q);
    check_and_free(array, dev, ctxt, usm::alloc::device);

    array = malloc_device<int>(
        N, q,
        property_list{
            ext::intel::experimental::property::usm::buffer_location(2)});
    check_and_free(array, dev, ctxt, usm::alloc::device);

    array = (int *)aligned_alloc_device(alignof(long long), N * sizeof(int),
                                        dev, ctxt);
    check_and_free(array, dev, ctxt, usm::alloc::device);

    array = (int *)aligned_alloc_device(alignof(long long), N * sizeof(int),
                                        dev, ctxt, property_list{});
    check_and_free(array, dev, ctxt, usm::alloc::device);
  }

  return 0;
}
