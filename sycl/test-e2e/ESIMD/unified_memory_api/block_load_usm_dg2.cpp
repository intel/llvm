//==------- block_load_usm_dg2.cpp - DPC++ ESIMD on-device test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu-intel-dg2

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The test verifies esimd::block_load() functions accepting USM pointer
// and optional compile-time esimd::properties.
// The block_load() calls in this test can use mask and cache-hint
// properties which require DG2 target device.

#include "Inputs/block_load.hpp"

int main() {
  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);

  constexpr auto TestFeatures = TestFeatures::DG2;
  bool Passed = true;

  Passed &= testUSM<int8_t, TestFeatures>(Q);
  Passed &= testUSM<int16_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp16))
    Passed &= testUSM<sycl::half, TestFeatures>(Q);
  Passed &= testUSM<uint32_t, TestFeatures>(Q);
  Passed &= testUSM<float, TestFeatures>(Q);
  Passed &= testUSM<int64_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp64))
    Passed &= testUSM<double, TestFeatures>(Q);

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
