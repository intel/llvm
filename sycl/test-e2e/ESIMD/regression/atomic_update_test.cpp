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
// TODO add support for atomic_load and atomic_store on esimd_emulator
// XFAIL: esimd_emulator
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>

int main() {
  sycl::queue q{};
  constexpr uint32_t Size = 16;
  auto int_sync = malloc_shared<uint32_t>(Size, q);
  auto fp_sync = malloc_shared<float>(Size, q);
  for (int i = 0; i < Size; ++i) {
    int_sync[i] = i;
    fp_sync[i] = i;
  }
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task([=]() [[intel::sycl_explicit_simd]] {
      __ESIMD_NS::simd<uint32_t, Size> offsets(0, sizeof(uint32_t));

      auto int_result =
          sycl::ext::intel::experimental::esimd::lsc_atomic_update<
              sycl::ext::intel::esimd::atomic_op::load, uint32_t, Size>(
              int_sync, offsets, 1);
      sycl::ext::intel::experimental::esimd::lsc_atomic_update<
          sycl::ext::intel::esimd::atomic_op::store, uint32_t, Size>(
          int_sync, offsets, int_result * 2, 1);
      auto fp_result = sycl::ext::intel::experimental::esimd::lsc_atomic_update<
          sycl::ext::intel::esimd::atomic_op::load, float, Size>(fp_sync,
                                                                 offsets, 1);
      sycl::ext::intel::experimental::esimd::lsc_atomic_update<
          sycl::ext::intel::esimd::atomic_op::store, float, Size>(
          fp_sync, offsets, fp_result * 2, 1);
    });
  });
  q.wait();

  bool passed = true;

  for (int i = 0; i < Size; ++i) {
    passed &= int_sync[i] == 2 * i;
    passed &= fp_sync[i] == 2 * i;
  }

  std::cout << (passed ? "Test passed\n" : "Test FAILED\n");

  free(int_sync, q);
  free(fp_sync, q);
  return passed ? 0 : 1;
}
