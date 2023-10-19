//==------- block_load_acc.cpp - DPC++ ESIMD on-device test ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The test verifies esimd::block_load() functions accepting ACCESSOR
// and optional compile-time esimd::properties.
// The block_load() calls in this test do not use mask or cache-hint
// properties to not impose using PVC features.

#include "Inputs/block_load.hpp"

int main() {
  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);

  constexpr bool TestPVCFeatures = true;
  bool Passed = true;

  Passed &= testACC<int8_t, !TestPVCFeatures>(Q);
  Passed &= testACC<int16_t, !TestPVCFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp16))
    Passed &= testACC<sycl::half, !TestPVCFeatures>(Q);
  Passed &= testACC<uint32_t, !TestPVCFeatures>(Q);
  Passed &= testACC<float, !TestPVCFeatures>(Q);
  Passed &=
      testACC<ext::intel::experimental::esimd::tfloat32, !TestPVCFeatures>(Q);
  Passed &= testACC<int64_t, !TestPVCFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp64))
    Passed &= testACC<double, !TestPVCFeatures>(Q);

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
