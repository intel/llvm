//==---------------- lsc_load_2d_u64.cpp - DPC++ ESIMD on-device test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "Inputs/lsc_load_prefetch_2d.hpp"

constexpr uint32_t seed = 322;
using T = uint64_t;

int main(void) {
  srand(seed);
  bool passed = true;

  // These parameters require unpadding. It is not implemented yet
  // passed &= test<0, T, 2, 2, 2, 2>(16, 4, 16, 1, 1);

  // non transposed, non transformed
  passed &= test<1, T, 1, 1, 8, 32>(8, 32, 8, 0, 0);
  passed &= test<2, T, 2, 2, 8, 4>(16, 16, 16, 1, 5);
  passed &= test<3, T, 1, 1, 4, 2>(16, 4, 16, 3, 1);

  // transposed
  passed &= test<4, T, 1, 1, 1, 8, 1, true>(16, 10, 16, 1, 2);
  passed &= test<5, T, 1, 1, 2, 8, 1, true>(16, 10, 16, 10, 1);
  passed &= test<6, T, 1, 1, 4, 8, 1, true>(16, 10, 16, 11, 1);
  passed &= test<7, T, 2, 2, 4, 8, 1, true>(16, 9, 16, 1, 1);

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
