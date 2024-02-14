//==------- slm_gather.cpp - DPC++ ESIMD on-device test --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Use per-kernel compilation to have more information about failing cases.
// RUN: %{build} -fsycl-device-code-split=per_kernel -D__ESIMD_GATHER_SCATTER_LLVM_IR -o %t.out
// RUN: %{run} %t.out

// The test verifies esimd::slm_gather() functions accepting  optional
// compile-time esimd::properties. The slm_gather() calls in this test do not
// use VS > 1 (number of loads per offset) to not impose using DG2/PVC features.

#include "Inputs/gather.hpp"

int main() {
  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);

  constexpr auto TestFeatures = TestFeatures::Generic;
  bool Passed = true;

  Passed &= testSLM<int8_t, TestFeatures>(Q);
  Passed &= testSLM<int16_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp16))
    Passed &= testSLM<sycl::half, TestFeatures>(Q);
  Passed &= testSLM<uint32_t, TestFeatures>(Q);
  Passed &= testSLM<float, TestFeatures>(Q);
  Passed &= testSLM<ext::intel::experimental::esimd::tfloat32, TestFeatures>(Q);
  Passed &= testSLM<int64_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp64))
    Passed &= testSLM<double, TestFeatures>(Q);
  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
