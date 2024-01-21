//==------- slm_gather_dg2_pvc.cpp - DPC++ ESIMD on-device test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-dg2 || gpu-intel-pvc
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// The test verifies esimd::slm_gather() functions accepting  optional
// compile-time esimd::properties. The slm_gather() calls in this test use
// VS > 1 (number of loads per offset) and require DG2 or PVC to run.

#include "Inputs/gather.hpp"

int main() {
  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);

  // DG2 and PVC support same gather() configurations. If some gather call
  // has corresponding instructions in PVC and does not have it in DG2, then
  // GPU RT emulates it for DG2.
  constexpr auto TestFeatures = TestFeatures::DG2;
  bool Passed = true;

  Passed &= testUSM<uint32_t, TestFeatures>(Q);
  Passed &= testUSM<float, TestFeatures>(Q);
  Passed &= testUSM<ext::intel::experimental::esimd::tfloat32, TestFeatures>(Q);
  Passed &= testUSM<int64_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp64))
    Passed &= testUSM<double, TestFeatures>(Q);

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
