//==--- copyto_copyfrom_acc_pvc.cpp - DPC++ ESIMD on-device test----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The test verifies copyto_copyfrom() functions accepting accessors
// and optional compile-time esimd::properties.
// The copyto_copyfrom() calls in this test use cache-hint
// properties which require PVC+ target device.

#include "Inputs/copyto_copyfrom.hpp"

int main() {
  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);

  constexpr auto TestFeatures = TestFeatures::PVC;
  bool Passed = true;

  Passed &= test_copyto_copyfrom_acc<int8_t, TestFeatures>(Q);
  Passed &= test_copyto_copyfrom_acc<int16_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp16))
    Passed &= test_copyto_copyfrom_acc<sycl::half, TestFeatures>(Q);
  Passed &= test_copyto_copyfrom_acc<uint32_t, TestFeatures>(Q);
  Passed &= test_copyto_copyfrom_acc<float, TestFeatures>(Q);
  Passed &= test_copyto_copyfrom_acc<ext::intel::experimental::esimd::tfloat32,
                                     TestFeatures>(Q);
  Passed &= test_copyto_copyfrom_acc<ext::intel::experimental::esimd::tfloat32,
                                     TestFeatures>(Q);
  Passed &= test_copyto_copyfrom_acc<int64_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp64))
    Passed &= test_copyto_copyfrom_acc<double, TestFeatures>(Q);

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
