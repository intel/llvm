//==------------ lsc_surf_prefetch_u16u32.cpp - DPC++ ESIMD on-device test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || esimd_emulator
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "Inputs/lsc_surf_load.hpp"

constexpr uint32_t seed = 197;
constexpr lsc_data_size DS = lsc_data_size::u16u32;

constexpr cache_hint L1H = cache_hint::cached;
constexpr cache_hint L3H = cache_hint::uncached;

int main(void) {
  srand(seed);
  bool passed = true;

  // non-transpose
  passed &= test<0, uint32_t, 1, 1, 1, 1, false, DS, L1H, L3H, true>(rand());
  passed &= test<1, uint32_t, 1, 4, 32, 1, false, DS, L1H, L3H, true>(rand());
  passed &= test<2, uint32_t, 2, 4, 16, 1, false, DS, L1H, L3H, true>(rand());
  passed &= test<3, uint32_t, 2, 2, 8, 1, false, DS, L1H, L3H, true>(rand());
  passed &= test<4, uint32_t, 4, 2, 4, 1, false, DS, L1H, L3H, true>(rand());
  passed &= test<5, uint32_t, 4, 16, 2, 1, false, DS, L1H, L3H, true>(rand());

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
