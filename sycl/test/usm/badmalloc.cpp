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
  assert(p == nullptr);

  // Bad size, host
  p = malloc(-1, q, usm::alloc::host);
  assert(p == nullptr);

  // Bad size, device
  p = malloc(-1, q, usm::alloc::device);
  assert(p == nullptr);

  // Bad size, shared
  p = malloc(-1, q, usm::alloc::shared);
  assert(p == nullptr);

  // Bad size, unknown
  p = malloc(-1, q, usm::alloc::unknown);
  assert(p == nullptr);

  return 0;
}
