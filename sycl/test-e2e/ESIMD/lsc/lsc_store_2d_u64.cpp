//==------------ lsc_store_2d_u64.cpp - DPC++ ESIMD on-device test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "Inputs/lsc_store_2d.hpp"

constexpr uint32_t seed = 363;
using T = uint64_t;

int main(void) {
  srand(seed);
  bool passed = true;

  passed &= test<1, T, 1, 1, 8, 8>(11, 20, 14, 3, 11);
  passed &= test<2, T, 2, 2, 2, 2>(3, 3, 8, 1, 1);

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
