//==---- fill.cpp - USM fill test ------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// XFAIL: cuda
// piextUSM*Alloc functions for CUDA are not behaving as described in
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t1.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out

#include <CL/sycl.hpp>

using namespace cl::sycl;

constexpr int count = 100;
constexpr int pattern = 42;

int main() {
  queue q;
  if (q.get_device().get_info<info::device::usm_shared_allocations>()) {
    int *mem = malloc_shared<int>(count, q);

    for (int i = 0; i < count; i++)
      mem[i] = 0;

    q.fill(mem, pattern, count);
    q.wait();

    for (int i = 0; i < count; i++) {
      assert(mem[i] == pattern);
    }
  }
  std::cout << "Passed\n";
  return 0;
}
