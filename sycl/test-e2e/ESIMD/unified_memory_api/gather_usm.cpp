//==------- gather_usm.cpp - DPC++ ESIMD on-device test --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The test verifies esimd::gather() functions accepting USM pointer
// and optional compile-time esimd::properties.
// The gather() calls in this test do not use cache-hint properties
// or VS > 1 (number of loads per offset) to not impose using PVC features.

#include "Inputs/gather.hpp"

int main() {
  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);

  constexpr bool TestPVCFeatures = true;
  bool Passed = true;

  Passed &= testUSM<int8_t, !TestPVCFeatures>(Q);
  Passed &= testUSM<int16_t, !TestPVCFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp16))
    Passed &= testUSM<sycl::half, !TestPVCFeatures>(Q);
  Passed &= testUSM<uint32_t, !TestPVCFeatures>(Q);
  Passed &= testUSM<float, !TestPVCFeatures>(Q);
  Passed &=
      testUSM<ext::intel::experimental::esimd::tfloat32, !TestPVCFeatures>(Q);
  Passed &= testUSM<int64_t, !TestPVCFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp64))
    Passed &= testUSM<double, !TestPVCFeatures>(Q);
  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
