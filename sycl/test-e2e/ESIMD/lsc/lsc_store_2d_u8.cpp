//==------------ lsc_store_2d_u8.cpp - DPC++ ESIMD on-device test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "Inputs/lsc_store_2d.hpp"

constexpr uint32_t seed = 336;
using T = uint8_t;

int main(void) {
  srand(seed);
  bool passed = true;

  passed &= test<1, T, 1, 1, 64, 8>(80, 20, 96, 16, 11);
  passed &= test<2, T, 2, 2, 4, 2>(16, 4, 64, 4, 1);

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
