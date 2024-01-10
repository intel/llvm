//==------- gather_acc.cpp - DPC++ ESIMD on-device test --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Use per-kernel compilation to have more information about failing cases.
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// The test verifies esimd::gather() functions accepting ACCESSOR
// and optional compile-time esimd::properties.
// The gather() calls in this test do not use cache-hint properties
// or VS > 1 (number of loads per offset) to not impose using PVC features.

#include "Inputs/gather.hpp"

int main() {
  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);

  constexpr auto TestFeatures = TestFeatures::Generic;
  bool Passed = true;

  Passed &= testACC<int8_t, TestFeatures>(Q);
  Passed &= testACC<int16_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp16))
    Passed &= testACC<sycl::half, TestFeatures>(Q);
  Passed &= testACC<uint32_t, TestFeatures>(Q);
  Passed &= testACC<float, TestFeatures>(Q);
  Passed &= testACC<ext::intel::experimental::esimd::tfloat32, TestFeatures>(Q);
#ifdef __ESIMD_FORCE_STATELESS_MEM
  Passed &= testACC<int64_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp64))
    Passed &= testACC<double, TestFeatures>(Q);
#endif // __ESIMD_FORCE_STATELESS_MEM
  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
