// RUN: %clangxx -fsycl %s -o %t1.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out

//==----------------- badmalloc.cpp - Bad Mallocs test ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


// This test verifies that things fail in the proper way when they should.

#include <CL/sycl.hpp>

using namespace cl::sycl;

int main(int argc, char *argv[]) {
  queue q;

  // Good size, bad type
  auto p = malloc(8, q, usm::alloc::unknown);
  if (p != nullptr)
    return 1;

  // Bad size, host
  p = malloc(-1, q, usm::alloc::host);
  if (p != nullptr)
    return 2;
  p = malloc(-1, q, usm::alloc::device);
  if (p != nullptr)
    return 3;
  p = malloc(-1, q, usm::alloc::shared);
  if (p != nullptr)
    return 4;
  p = malloc(-1, q, usm::alloc::unknown);
  if (p != nullptr)
    return 5;

  // Bad size, auto aligned
  p = aligned_alloc(0, -1, q,  usm::alloc::host);
  if (p != nullptr)
    return 6;
  p = aligned_alloc(0, -1, q,  usm::alloc::device);
  if (p != nullptr)
    return 7;
  p = aligned_alloc(0, -1, q,  usm::alloc::shared);
  if (p != nullptr)
    return 8;
  p = aligned_alloc(0, -1, q,  usm::alloc::unknown);
  if (p != nullptr)
    return 9;

  // Allocs of 0 undefined, but bad type
  p = aligned_alloc(4, 0, q,  usm::alloc::unknown);
  if (p != nullptr)
    return 10;

  return 0;
}
