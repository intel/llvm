// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out

//==---------- allocator_equal.cpp - Allocator Equality test ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <cassert>

using namespace sycl;

int main() {
  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();

  queue q2;
  auto dev2 = q2.get_device();
  auto ctxt2 = q2.get_context();

  // Test allocator equality
  if (dev.get_info<info::device::usm_host_allocations>()) {
    usm_allocator<int, usm::alloc::host> alloc1(ctxt, dev);
    usm_allocator<int, usm::alloc::host> alloc2(q);

    assert((alloc1 == alloc2) && "Allocators should be equal.");

    usm_allocator<int, usm::alloc::host, 8> alloc3(ctxt, dev);
    usm_allocator<int, usm::alloc::host, 16> alloc4(q);

    assert((alloc1 == alloc2) && "Allocators should be equal.");
  }

  if (dev.get_info<info::device::usm_shared_allocations>() &&
      dev.get_info<info::device::usm_host_allocations>()) {
    usm_allocator<int, usm::alloc::shared> alloc1(ctxt, dev);
    usm_allocator<int, usm::alloc::host> alloc2(ctxt, dev);

    assert((alloc1 != alloc2) && "Allocators should NOT be equal.");
  }

  return 0;
}
