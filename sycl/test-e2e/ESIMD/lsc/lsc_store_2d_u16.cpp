//==------------ lsc_store_2d_u16.cpp - DPC++ ESIMD on-device test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc
// TODO: GPU Driver fails with "add3 src operand only supports integer D/W type"
// error. Enable the test when it is fixed.
// UNSUPPORTED: gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "Inputs/lsc_store_2d.hpp"

constexpr uint32_t seed = 295;
using T = uint16_t;

int main(void) {
  srand(seed);
  bool passed = true;

  passed &= test<1, T, 1, 1, 32, 8>(40, 20, 64, 8, 11);
  passed &= test<2, T, 2, 2, 2, 2>(16, 4, 32, 2, 1);

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
