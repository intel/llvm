//==------------ lsc_block_prefetch_u16.cpp - DPC++ ESIMD on-device test ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "Inputs/lsc_block_load.hpp"

constexpr uint32_t seed = 322;
using T = uint16_t;

constexpr cache_hint L1H = cache_hint::cached;
constexpr cache_hint L3H = cache_hint::uncached;

int main(void) {
  srand(seed);
  bool passed = true;

  // These parameters require unpadding. It is not implemented yet
  // passed &= test<0, T, 2, 2, 2, 2>(16, 4, 16, 1, 1);

  // non transposed, non transformed
  passed &= test<1, T, 1, 1, 16, 32, 1, false, false, L1H, L3H, true>(
      24, 64, 64, 6, 21);
  passed &=
      test<2, T, 2, 2, 8, 4, 1, false, false, L1H, L3H, true>(16, 16, 32, 2, 5);
  passed &=
      test<3, T, 1, 1, 8, 4, 2, false, false, L1H, L3H, true>(16, 7, 32, 4, 1);

  // transformed
  passed &=
      test<4, T, 1, 1, 8, 4, 4, false, true, L1H, L3H, true>(16, 10, 32, 6, 5);
  passed &= test<5, T, 1, 1, 6, 10, 1, false, true, L1H, L3H, true>(18, 10, 32,
                                                                    12, 0);
  passed &=
      test<6, T, 1, 1, 16, 2, 2, false, true, L1H, L3H, true>(32, 4, 32, 4, 1);
  passed &=
      test<7, T, 2, 2, 2, 16, 2, false, true, L1H, L3H, true>(4, 20, 32, 0, 3);
  passed &= test<8, T, 1, 1, 16, 32, 1, false, true, L1H, L3H, true>(24, 50, 32,
                                                                     4, 14);

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
