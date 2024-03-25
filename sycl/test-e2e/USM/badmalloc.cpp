// UNSUPPORTED: windows
//
// RUN: %{build} -o %t1.out
// RUN: %{run} %t1.out
// UNSUPPORTED: ze_debug

//==----------------- badmalloc.cpp - Bad Mallocs test ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test verifies that things fail in the proper way when they should.

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

int main(int argc, char *argv[]) {
  queue q;

  // Good size, bad type
  auto p = malloc(8, q, usm::alloc::unknown);
  if (p != nullptr)
    return 1;
  // check that malloc_shared throws when usm_shared_allocations not supported
  if (!q.get_device().has(aspect::usm_shared_allocations)) {
    try {
      auto p = malloc_shared<int>(1, q);
      return 11;
    } catch (const sycl::exception &e) {
      if (e.code() != sycl::errc::feature_not_supported)
        return 11;
    }
  }
  // Bad size, host
  p = malloc(-1, q, usm::alloc::host);
  std::cout << "p = " << p << std::endl;
  if (p != nullptr)
    return 2;
  p = malloc(-1, q, usm::alloc::device);
  std::cout << "p = " << p << std::endl;
  if (p != nullptr)
    return 3;
  if (q.get_device().has(aspect::usm_shared_allocations)) {
    p = malloc(-1, q, usm::alloc::shared);
    std::cout << "p = " << p << std::endl;
    if (p != nullptr)
      return 4;
  }
  p = malloc(-1, q, usm::alloc::unknown);
  std::cout << "p = " << p << std::endl;
  if (p != nullptr)
    return 5;

  // Bad size, auto aligned
  p = aligned_alloc(0, -1, q, usm::alloc::host);
  std::cout << "p = " << p << std::endl;
  if (p != nullptr)
    return 6;
  p = aligned_alloc(0, -1, q, usm::alloc::device);
  std::cout << "p = " << p << std::endl;
  if (p != nullptr)
    return 7;
  if (q.get_device().has(aspect::usm_shared_allocations)) {
    p = aligned_alloc(0, -1, q, usm::alloc::shared);
    std::cout << "p = " << p << std::endl;
    if (p != nullptr)
      return 8;
  }
  p = aligned_alloc(0, -1, q, usm::alloc::unknown);
  std::cout << "p = " << p << std::endl;
  if (p != nullptr)
    return 9;

  // Allocs of 0 undefined, but bad type
  p = aligned_alloc(4, 0, q, usm::alloc::unknown);
  std::cout << "p = " << p << std::endl;
  if (p != nullptr)
    return 10;

  return 0;
}
