//==------- lsc_usm_gather_u64.cpp - DPC++ ESIMD on-device test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc || gpu-intel-dg2
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "Inputs/lsc_usm_gather_prefetch.hpp"

int main(void) {
  constexpr uint32_t Seed = 185;
  auto Q = queue{gpu_selector_v};
  srand(Seed);

  bool Passed = true;
  Passed &= test_lsc_gather<uint64_t>();
  if (Q.get_device().has(sycl::aspect::fp64)) {
    Passed &= test_lsc_gather<double>();
  }
  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
