// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: %RUN_ON_HOST %t1.out

//==---------- allocator_equal.cpp - Allocator Equality test ---------------==//
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

  {
    // Test usm_allocator
    if (dev.get_info<info::device::usm_shared_allocations>() &&
        dev.get_info<info::device::usm_host_allocations>()) {
      usm_allocator<int, usm::alloc::shared> alloc11(ctxt, dev);
      usm_allocator<int, usm::alloc::shared> alloc12(ctxt, dev);
      usm_allocator<int, usm::alloc::host> alloc21(q);
      usm_allocator<int, usm::alloc::host> alloc22(alloc21);

      // usm::alloc::device is not supported by usm_allocator

      assert((alloc11 != alloc22) && "Allocators should NOT be equal.");
      assert((alloc11 == alloc12) && "Allocators should be equal.");
      assert((alloc21 == alloc22) && "Allocators should be equal.");
    }
  }

  {
    // Test use of allocator in containers
    if (dev.get_info<info::device::usm_shared_allocations>()) {
      usm_allocator<int, usm::alloc::shared> alloc(ctxt, dev);

      std::vector<int, decltype(alloc)> vec(alloc);
      vec.resize(N);

      for (int i = 0; i < N; i++) {
        vec[i] = i;
      }

      int *vals = &vec[0];

      q.submit([=](handler &h) {
         h.single_task<class bar>([=]() {
           for (int i = 1; i < N; i++) {
             vals[0] += vals[i];
           }
         });
       }).wait();

      if (vals[0] != ((N * (N - 1)) / 2))
        return -1;
    }
  }

  {

    // Test utility functions
    if (dev.get_info<info::device::usm_shared_allocations>() &&
        dev.get_info<info::device::usm_host_allocations>() &&
        dev.get_info<info::device::usm_device_allocations>()) {
      int *array;
      array = (int *)malloc_host(N * sizeof(int), q);
      assert((get_pointer_type(array, ctxt) == usm::alloc::host) &&
             "Allcoation pointer should be host type");
      assert((get_pointer_device(array, ctxt) == dev) &&
             "Allcoation pointer should be host type");
      free(array, ctxt);

      array = (int *)malloc_shared(N * sizeof(int), q);
      // host device treats all allocations as host allocations
      assert((get_pointer_type(array, ctxt) == usm::alloc::host) &&
             "Allcoation pointer should be host type");
      assert((get_pointer_device(array, ctxt) == dev) &&
             "Allcoation pointer should be host type");
      free(array, ctxt);

      array = (int *)malloc_device(N * sizeof(int), q);
      // host device treats all allocations as host allocations
      assert((get_pointer_type(array, ctxt) == usm::alloc::host) &&
             "Allcoation pointer should be host type");
      assert((get_pointer_device(array, ctxt) == dev) &&
             "Allcoation pointer should be host type");
      free(array, ctxt);

      // Test aligned allocation
      array =
          (int *)aligned_alloc_host(alignof(long long), N * sizeof(int), ctxt);
      assert((get_pointer_type(array, ctxt) == usm::alloc::host) &&
             "Allcoation pointer should be host type");
      assert((get_pointer_device(array, ctxt) == dev) &&
             "Allcoation pointer should be host type");
      free(array, ctxt);

      array = (int *)aligned_alloc_shared(alignof(long long), N * sizeof(int),
                                          dev, ctxt);
      // host device treats all allocations as host allocations
      assert((get_pointer_type(array, ctxt) == usm::alloc::host) &&
             "Allcoation pointer should be host type");
      assert((get_pointer_device(array, ctxt) == dev) &&
             "Allcoation pointer should be host type");
      free(array, ctxt);

      array = (int *)aligned_alloc_device(alignof(long long), N * sizeof(int),
                                          dev, ctxt);
      // host device treats all allocations as host allocations
      assert((get_pointer_type(array, ctxt) == usm::alloc::host) &&
             "Allcoation pointer should be host type");
      assert((get_pointer_device(array, ctxt) == dev) &&
             "Allcoation pointer should be host type");
      free(array, ctxt);
    }
  }
  return 0;
}
