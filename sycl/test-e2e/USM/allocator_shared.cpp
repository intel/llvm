// RUN: %{build} -o %t1.out
// RUN: %{run} %t1.out

//==-------- allocator_shared.cpp - Allocate Shared test -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>
#include <sycl/usm/usm_allocator.hpp>

#include <cassert>
#include <memory>

using namespace sycl;

int main() {
  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();

  // Test ability to create a shared pointer.
  if (dev.get_info<info::device::usm_host_allocations>()) {
    usm_allocator<int, usm::alloc::host> alloc(ctxt, dev);
    auto ptr1 = std::allocate_shared<int>(alloc);

    // Test construction
    auto ptr2 = std::allocate_shared<int>(alloc, 42);
    assert((*ptr2 == 42) && "Host construct passed.");
  }

  if (dev.get_info<info::device::usm_shared_allocations>()) {
    usm_allocator<int, usm::alloc::shared> alloc(ctxt, dev);
    auto ptr1 = std::allocate_shared<int>(alloc);

    // Test construction
    auto ptr2 = std::allocate_shared<int>(alloc, 42);
    assert((*ptr2 == 42) && "Shared construct passed.");
  }

  // Device allocations are not supported due to how allocated_shared is
  // written.

  return 0;
}
