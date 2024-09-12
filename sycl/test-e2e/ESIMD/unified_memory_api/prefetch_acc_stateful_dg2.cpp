//==-- prefetch_acc_stateful_dg2.cpp - DPC++ ESIMD on-device test ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-dg2 || arch-intel_gpu_pvc
// RUN: %{build} -fsycl-device-code-split=per_kernel -fno-sycl-esimd-force-stateless-mem -o %t.out
// RUN: %{run} %t.out

// The test verifies esimd::prefetch() functions accepting accessor
// and optional compile-time esimd::properties.
// The prefetch() calls in this test require DG2 or PVC to run.

#include "Inputs/prefetch.hpp"

int main() {
  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);
  constexpr auto TestFeatures = TestFeatures::DG2;

  bool Passed = true;

  Passed &= testACC<int8_t>(Q);
  Passed &= testACC<int16_t>(Q);
  if (Q.get_device().has(sycl::aspect::fp16))
    Passed &= testACC<sycl::half>(Q);
  Passed &= testACC<uint32_t>(Q);
  Passed &= testACC<float>(Q);
  Passed &= testACC<ext::intel::experimental::esimd::tfloat32>(Q);
  Passed &= testACC<int64_t>(Q);
  if (Q.get_device().has(sycl::aspect::fp64))
    Passed &= testACC<double>(Q);

  Passed &= testBlockLoadPrefetchACC<int8_t, TestFeatures>(Q);
  Passed &= testBlockLoadPrefetchACC<int16_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp16))
    Passed &= testBlockLoadPrefetchACC<sycl::half, TestFeatures>(Q);
  Passed &= testBlockLoadPrefetchACC<uint32_t, TestFeatures>(Q);
  Passed &= testBlockLoadPrefetchACC<float, TestFeatures>(Q);
  Passed &= testBlockLoadPrefetchACC<ext::intel::experimental::esimd::tfloat32,
                                     TestFeatures>(Q);
  Passed &= testBlockLoadPrefetchACC<int64_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp64))
    Passed &= testBlockLoadPrefetchACC<double, TestFeatures>(Q);

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
