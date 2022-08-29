//==------------ lsc_block_store_u16.cpp - DPC++ ESIMD on-device test ------==//
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

#include "Inputs/lsc_block_store.hpp"

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
