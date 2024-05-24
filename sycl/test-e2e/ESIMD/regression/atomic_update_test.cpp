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
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "../esimd_test_utils.hpp"

int main() {
  sycl::queue q{};
  auto int_sync = malloc_shared<uint32_t>(16, q);
  auto fp_sync = malloc_shared<float>(16, q);
  int_sync[0] = 5;
  int_sync[1] = 0;
  fp_sync[0] = 6;
  fp_sync[1] = 0;
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class reproducer>([=]() [[intel::sycl_explicit_simd]] {
      uint32_t int_result =
          sycl::ext::intel::experimental::esimd::lsc_atomic_update<
              sycl::ext::intel::esimd::atomic_op::load, uint32_t, 1>(int_sync,
                                                                     0, 1)[0];
      sycl::ext::intel::experimental::esimd::lsc_atomic_update<
          sycl::ext::intel::esimd::atomic_op::store, uint32_t, 1>(
          int_sync, sizeof(uint32_t), int_result, 1);
      float fp_result =
          sycl::ext::intel::experimental::esimd::lsc_atomic_update<
              sycl::ext::intel::esimd::atomic_op::load, float, 1>(fp_sync, 0,
                                                                  1)[0];
      sycl::ext::intel::experimental::esimd::lsc_atomic_update<
          sycl::ext::intel::esimd::atomic_op::store, float, 1>(
          fp_sync, sizeof(float), fp_result, 1);
    });
  });
  q.wait();

  bool passed = true;

  passed &= int_sync[0] == int_sync[1];
  passed &= fp_sync[0] == fp_sync[1];

  std::cout << (passed ? "Test passed\n" : "Test FAILED\n");

  free(int_sync, q);
  free(fp_sync, q);
  return passed ? 0 : 1;
}
