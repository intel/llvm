//==------- prefetch_usm_pvc.cpp - DPC++ ESIMD on-device test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// The test verifies esimd::prefetch() functions accepting USM pointer
// and optional compile-time esimd::properties.
// The prefetch() calls in this test require PVC to run.

#include "Inputs/prefetch.hpp"

int main() {
  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);
  constexpr auto TestFeatures = TestFeatures::PVC;

  bool Passed = true;

  Passed &= testUSM<int8_t>(Q);
  Passed &= testUSM<int16_t>(Q);
  if (Q.get_device().has(sycl::aspect::fp16))
    Passed &= testUSM<sycl::half>(Q);
  Passed &= testUSM<uint32_t>(Q);
  Passed &= testUSM<float>(Q);
  Passed &= testUSM<ext::intel::experimental::esimd::tfloat32>(Q);
  Passed &= testUSM<int64_t>(Q);
  if (Q.get_device().has(sycl::aspect::fp64))
    Passed &= testUSM<double>(Q);

  Passed &= testBlockLoadPrefetchUSM<int8_t, TestFeatures>(Q);
  Passed &= testBlockLoadPrefetchUSM<int16_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp16))
    Passed &= testBlockLoadPrefetchUSM<sycl::half, TestFeatures>(Q);
  Passed &= testBlockLoadPrefetchUSM<uint32_t, TestFeatures>(Q);
  Passed &= testBlockLoadPrefetchUSM<float, TestFeatures>(Q);
  Passed &= testBlockLoadPrefetchUSM<ext::intel::experimental::esimd::tfloat32,
                                     TestFeatures>(Q);
  Passed &= testBlockLoadPrefetchUSM<int64_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp64))
    Passed &= testBlockLoadPrefetchUSM<double, TestFeatures>(Q);

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}