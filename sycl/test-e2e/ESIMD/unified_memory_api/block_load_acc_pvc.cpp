//==------- block_load_acc_pvc.cpp - DPC++ ESIMD on-device test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu-intel-pvc

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The test verifies esimd::block_load() functions accepting USM pointer
// and optional compile-time esimd::properties.
// The block_load() calls in this test can use mask and cache-hint
// properties which require PVC+ target device.

#include "Inputs/block_load.hpp"

int main() {
  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);

  constexpr auto TestFeatures = TestFeatures::PVC;
  bool Passed = true;

  Passed &= testACC<int8_t, TestFeatures>(Q);
  Passed &= testACC<int16_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp16))
    Passed &= testACC<sycl::half, TestFeatures>(Q);
  Passed &= testACC<uint32_t, TestFeatures>(Q);
  Passed &= testACC<float, TestFeatures>(Q);
  Passed &= testACC<ext::intel::experimental::esimd::tfloat32, TestFeatures>(Q);
  Passed &= testACC<int64_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp64))
    Passed &= testACC<double, TestFeatures>(Q);

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
