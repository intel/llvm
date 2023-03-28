//==------- lsc_usm_gather_u32.cpp - DPC++ ESIMD on-device test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "Inputs/lsc_usm_gather_prefetch.hpp"

int main(void) {
  constexpr uint32_t Seed = 185;
  srand(Seed);

  bool Passed = true;
  Passed &= test_lsc_gather<uint32_t>();
  Passed &= test_lsc_gather<float>();
  Passed &= test_lsc_gather<sycl::ext::intel::experimental::esimd::tfloat32>();

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
