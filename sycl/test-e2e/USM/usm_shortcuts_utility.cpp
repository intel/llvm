// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==------ usm_shortcuts_utility.cpp - USM shortcuts test ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>

#include <sycl/ext/intel/experimental/usm_properties.hpp>
#include <sycl/usm.hpp>

#include <cassert>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

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

  if (dev.get_info<sycl::info::device::usm_host_allocations>()) {
    array = (int *)malloc(N * sizeof(int), dev, usm::alloc::host);
    check_and_free(array, dev, ctxt, usm::alloc::host);

    array =
        (int *)malloc(N * sizeof(int), dev, usm::alloc::host, property_list{});
    check_and_free(array, dev, ctxt, usm::alloc::host);

    array = malloc<int>(N * sizeof(int), dev, usm::alloc::host);
    check_and_free(array, dev, ctxt, usm::alloc::host);

    array =
        malloc<int>(N * sizeof(int), dev, usm::alloc::host, property_list{});
    check_and_free(array, dev, ctxt, usm::alloc::host);

    array = (int *)aligned_alloc(alignof(long long), N * sizeof(int), dev,
                                 usm::alloc::host);
    check_and_free(array, dev, ctxt, usm::alloc::host);

    array = (int *)aligned_alloc(alignof(long long), N * sizeof(int), dev,
                                 usm::alloc::host, property_list{});
    check_and_free(array, dev, ctxt, usm::alloc::host);

    array = aligned_alloc<int>(alignof(long long), N * sizeof(int), dev,
                               usm::alloc::host);
    check_and_free(array, dev, ctxt, usm::alloc::host);

    array = aligned_alloc<int>(alignof(long long), N * sizeof(int), dev,
                               usm::alloc::host, property_list{});
    check_and_free(array, dev, ctxt, usm::alloc::host);
  }

  if (dev.get_info<sycl::info::device::usm_shared_allocations>()) {
    array = (int *)malloc_shared(N * sizeof(int), dev);
    check_and_free(array, dev, ctxt, usm::alloc::shared);

    array = (int *)malloc_shared(N * sizeof(int), dev, property_list{});
    check_and_free(array, dev, ctxt, usm::alloc::shared);

    array = malloc_shared<int>(N * sizeof(int), dev);
    check_and_free(array, dev, ctxt, usm::alloc::shared);

    array = malloc_shared<int>(N * sizeof(int), dev, property_list{});
    check_and_free(array, dev, ctxt, usm::alloc::shared);

    array =
        (int *)aligned_alloc_shared(alignof(long long), N * sizeof(int), dev);
    check_and_free(array, dev, ctxt, usm::alloc::shared);

    array = (int *)aligned_alloc_shared(alignof(long long), N * sizeof(int),
                                        dev, property_list{});
    check_and_free(array, dev, ctxt, usm::alloc::shared);

    array = aligned_alloc_shared<int>(alignof(long long), N * sizeof(int), dev);
    check_and_free(array, dev, ctxt, usm::alloc::shared);

    array = aligned_alloc_shared<int>(alignof(long long), N * sizeof(int), dev,
                                      property_list{});
    check_and_free(array, dev, ctxt, usm::alloc::shared);
  }

  if (dev.get_info<sycl::info::device::usm_device_allocations>()) {
    array = (int *)malloc_device(N * sizeof(int), dev);
    check_and_free(array, dev, ctxt, usm::alloc::device);

    array = (int *)malloc_device(N, dev, property_list{});
    check_and_free(array, dev, ctxt, usm::alloc::device);

    array = malloc_device<int>(N * sizeof(int), dev);
    check_and_free(array, dev, ctxt, usm::alloc::device);

    array = malloc_device<int>(N, dev, property_list{});
    check_and_free(array, dev, ctxt, usm::alloc::device);

    array =
        (int *)aligned_alloc_device(alignof(long long), N * sizeof(int), dev);
    check_and_free(array, dev, ctxt, usm::alloc::device);

    array = (int *)aligned_alloc_device(alignof(long long), N * sizeof(int),
                                        dev, property_list{});
    check_and_free(array, dev, ctxt, usm::alloc::device);

    array = aligned_alloc_device<int>(alignof(long long), N * sizeof(int), dev);
    check_and_free(array, dev, ctxt, usm::alloc::device);

    array = aligned_alloc_device<int>(alignof(long long), N * sizeof(int), dev,
                                      property_list{});
    check_and_free(array, dev, ctxt, usm::alloc::device);
  }

  return 0;
}
