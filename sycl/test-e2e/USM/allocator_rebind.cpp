// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out

//==------------------------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;

// Allocator has a minimum internal alignment of 64, so use higher values for
// the purpose of this test.
struct alignas(256) Aligned256 {
  int x;
};
struct alignas(128) Aligned128 {
  int x;
};

template <class Allocator>
void test(Allocator alloc, size_t Align, int Line = __builtin_LINE()) {
  decltype(alloc.allocate(1)) Ptrs[10];
  for (auto *&Elem : Ptrs)
    Elem = alloc.allocate(1);

  for (auto *Ptr : Ptrs) {
    if ((reinterpret_cast<uintptr_t>(Ptr) & (Align - 1)) != 0) {
      std::cout << "Failed at line " << Line << std::endl;
      assert(false && "Not properly aligned!");
    }
  }

  for (auto *Ptr : Ptrs)
    alloc.deallocate(Ptr, 1);
}

int main() {
  queue q;
  auto dev = q.get_device();
  auto ctx = q.get_context();

  if (!dev.get_info<info::device::usm_host_allocations>())
    return 0;

  // There was a bug that we didn't re-adjust allocator's alignment when we
  // rebound it to another type.
  {
    // Test default value of Alignment template parameter.
    usm_allocator<Aligned128, usm::alloc::host> alloc(ctx, dev);
    test(alloc, 128);

    using traits_t = std::allocator_traits<decltype(alloc)>;
    using rebind_t = typename traits_t::template rebind_alloc<Aligned256>;
    rebind_t alloc_rebound = alloc;
    test(alloc_rebound, 256);
  }

  {
    // Test explicit value of Alignment template parameter.
    usm_allocator<Aligned256, usm::alloc::host, 256> alloc(ctx, dev);
    test(alloc, 256);
    using traits_t = std::allocator_traits<decltype(alloc)>;
    using rebind_t = typename traits_t::template rebind_alloc<Aligned128>;
    rebind_t alloc_rebound = alloc;
    // Rebound allocator must use the higher alignment from the explicit
    // template parameter, not the type's alignment.
    test(alloc_rebound, 256);
  }

  return 0;
}
