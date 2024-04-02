//==------- prefetch_usm_dg2_pvc.cpp - DPC++ ESIMD on-device test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-dg2 || gpu-intel-pvc
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// The test verifies esimd::prefetch() functions accepting USM pointer
// and optional compile-time esimd::properties.
// The prefetch() calls in this test require DG2 or PVC to run.

#include "Inputs/prefetch.hpp"

int main() {
  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);

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

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
