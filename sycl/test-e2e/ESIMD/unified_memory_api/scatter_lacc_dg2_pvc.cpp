//==------- scatter_lacc_dg2_pvc.cpp - DPC++ ESIMD on-device test------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || gpu-intel-dg2
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// The test verifies esimd::scatter() functions accepting local accessor
// and optional compile-time esimd::properties.
// The scatter() calls in this test use VS > 1 (number of loads per
// offset) and requires DG2 or PVC.

#include "Inputs/scatter.hpp"

int main() {
  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);

  constexpr auto TestFeatures = TestFeatures::PVC;
  bool Passed = true;

  Passed &= testLACC<int8_t, TestFeatures>(Q);
  Passed &= testLACC<int16_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp16))
    Passed &= testLACC<sycl::half, TestFeatures>(Q);
  Passed &= testLACC<uint32_t, TestFeatures>(Q);
  Passed &= testLACC<float, TestFeatures>(Q);
  Passed &= testLACC<int64_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp64))
    Passed &= testLACC<double, TestFeatures>(Q);

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
