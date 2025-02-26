//==------- gather_lacc_dg2_pvc.cpp - DPC++ ESIMD on-device test ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu-intel-dg2 || arch-intel_gpu_pvc
// REQUIRES-INTEL-DRIVER: lin: 26690, win: 101.4576

// Use per-kernel compilation to have more information about failing cases.
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// The test verifies esimd::gather() functions accepting Local ACCESSOR
// and optional compile-time esimd::properties.
// The gather() calls in this test can use  VS > 1 (number of loads per offset).

#include "Inputs/gather.hpp"

int main() {
  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);

  constexpr auto TestFeatures = TestFeatures::DG2;
  bool Passed = true;

  Passed &= testLACC<int8_t, TestFeatures>(Q);
  Passed &= testLACC<int16_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp16))
    Passed &= testLACC<sycl::half, TestFeatures>(Q);
  Passed &= testLACC<uint32_t, TestFeatures>(Q);
  Passed &= testLACC<float, TestFeatures>(Q);
  Passed &=
      testLACC<ext::intel::experimental::esimd::tfloat32, TestFeatures>(Q);

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
