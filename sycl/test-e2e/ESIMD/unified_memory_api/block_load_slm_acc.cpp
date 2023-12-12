//==------- block_load_slm_acc.cpp - DPC++ ESIMD on-device test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES-INTEL-DRIVER: lin: 27202, win: 101.4677
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The test verifies esimd::block_load() functions accepting local accessor and
// using optional compile-time esimd::properties.
// The block_load() calls in this test do not use the mask operand and do not
// require PVC features.

#include "Inputs/block_load.hpp"

int main() {
  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);

  constexpr auto TestFeatures = TestFeatures::Generic;
  bool Passed = true;

  Passed &= testSLMAcc<int8_t, TestFeatures>(Q);
  Passed &= testSLMAcc<int16_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp16))
    Passed &= testSLMAcc<sycl::half, TestFeatures>(Q);
  Passed &= testSLMAcc<uint32_t, TestFeatures>(Q);
  Passed &= testSLMAcc<float, TestFeatures>(Q);
  Passed &=
      testSLMAcc<ext::intel::experimental::esimd::tfloat32, TestFeatures>(Q);
  Passed &= testSLMAcc<int64_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp64))
    Passed &= testSLMAcc<double, TestFeatures>(Q);

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
