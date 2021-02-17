// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: %RUN_ON_HOST %t1.out

//==--------- usm_allocator.cpp - USM allocator construction test ----------==//
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

  return 0;
}
