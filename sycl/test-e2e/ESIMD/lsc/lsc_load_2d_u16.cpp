//==---------------- lsc_load_2d_u16.cpp - DPC++ ESIMD on-device test ------==//
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
using T = uint16_t;

int main(void) {
  srand(seed);
  bool passed = true;

  // non transposed, non transformed
  passed &= test<1, T, 1, 1, 16, 32>(24, 64, 64, 6, 21);
  passed &= test<2, T, 2, 2, 8, 4>(16, 16, 32, 2, 5);
  passed &= test<3, T, 1, 1, 8, 4, 2>(16, 7, 32, 4, 1);

  // transformed
  passed &= test<4, T, 1, 1, 6, 10, 1, false, true>(18, 10, 32, 12, 0);
  passed &= test<5, T, 1, 1, 8, 4, 4, false, true>(16, 10, 32, 6, 5);
  passed &= test<6, T, 1, 1, 16, 2, 2, false, true>(32, 4, 32, 4, 1);
  passed &= test<7, T, 1, 1, 2, 16, 2, false, true>(4, 20, 32, 0, 3);
  passed &= test<8, T, 1, 1, 16, 32, 1, false, true>(24, 50, 32, 4, 14);
  passed &= test<9, T, 1, 1, 6, 4, 4, false, true>(32, 10, 32, 4, 0);
  passed &= test<10, T, 1, 1, 6, 4, 2, false, true>(16, 10, 32, 4, 0);
  passed &= test<11, T, 1, 1, 4, 8, 2, false, true>(16, 10, 32, 4, 0);
  passed &= test<12, T, 1, 1, 2, 16, 4, false, true>(16, 10, 32, 4, 0);

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
