//==------- lsc_usm_gather_u16u32.cpp - DPC++ ESIMD on-device test ---------==//
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
  constexpr uint32_t Seed = 186;
  constexpr lsc_data_size DS = lsc_data_size::u16u32;
  srand(Seed);

  bool Passed = true;
  Passed &= test_lsc_gather<0, uint32_t, DS>();

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
