//==---------------- atomic_update_test.cpp  - DPC++ ESIMD on-device test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test checks regression in lsc_atomic_update
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>

int main() {
  sycl::queue q{};
  auto p_sync = malloc_shared<uint32_t>(1024, q);
  p_sync[0] = 5;
  p_sync[1] = 0;
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class reproducer>([=]() [[intel::sycl_explicit_simd]] {
      uint32_t result =
          sycl::ext::intel::experimental::esimd::lsc_atomic_update<
              sycl::ext::intel::esimd::atomic_op::load, uint32_t, 1>(p_sync, 0,
                                                                     1)[0];
      sycl::ext::intel::experimental::esimd::lsc_atomic_update<
          sycl::ext::intel::esimd::atomic_op::store, uint32_t, 1>(p_sync, 1,
                                                                  result, 1);
    });
  });
  q.wait();

  bool passed = true;

  passed &= p_sync[0] == p_sync[1];

  std::cout << (passed ? "Test passed\n" : "Test FAILED\n");

  free(p_sync, q);
  return passed ? 0 : 1;
}