//==---------------- lsc_load_2d_u8.cpp - DPC++ ESIMD on-device test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "Inputs/lsc_load_prefetch_2d.hpp"

constexpr uint32_t seed = 322;
using T = uint8_t;

int main(void) {
  srand(seed);
  bool passed = true;

  // non transposed, non transformed
  passed &= test<1, T, 1, 1, 16, 32, 2>(40, 64, 64, 4, 21);
  passed &= test<2, T, 2, 2, 8, 8, 2>(16, 16, 64, 8, 5);
  passed &= test<3, T, 1, 1, 8, 32, 2>(16, 80, 64, 4, 1);

  // transformed
  passed &= test<4, T, 1, 1, 16, 4, 4, false, true>(100, 10, 128, 16, 5);
  passed &= test<5, T, 1, 1, 12, 20, 1, false, true>(16, 40, 64, 0, 0);
  passed &= test<6, T, 1, 1, 16, 4, 2, false, true>(32, 4, 64, 4, 1);
  passed &= test<7, T, 2, 2, 4, 16, 2, false, true>(4, 20, 64, 0, 3);
  passed &= test<8, T, 1, 1, 16, 32, 1, false, true>(24, 80, 64, 4, 14);
  passed &= test<9, T, 1, 1, 16, 4, 4, false, true>(64, 10, 64, 0, 0);

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
