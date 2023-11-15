//==--- block_store_acc_pvc.cpp - DPC++ ESIMD on-device test----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The test verifies esimd::block_store() functions accepting accessors
// and optional compile-time esimd::properties.
// The block_store() calls in this test use cache-hint
// properties which require PVC+ target device.

#include "Inputs/block_store.hpp"

int main() {
  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);

  constexpr bool TestPVCFeatures = true;
  bool Passed = true;

  Passed &= test_block_store_acc<int8_t, TestPVCFeatures>(Q);
  Passed &= test_block_store_acc<int16_t, TestPVCFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp16))
    Passed &= test_block_store_acc<sycl::half, TestPVCFeatures>(Q);
  Passed &= test_block_store_acc<uint32_t, TestPVCFeatures>(Q);
  Passed &= test_block_store_acc<float, TestPVCFeatures>(Q);
  Passed &= test_block_store_acc<ext::intel::experimental::esimd::tfloat32,
                                 TestPVCFeatures>(Q);
  Passed &= test_block_store_acc<ext::intel::experimental::esimd::tfloat32,
                                 !TestPVCFeatures>(Q);
  Passed &= test_block_store_acc<int64_t, TestPVCFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp64))
    Passed &= test_block_store_acc<double, TestPVCFeatures>(Q);

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
