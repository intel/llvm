//==------- lsc_usm_store_u8_u16.cpp - DPC++ ESIMD on-device test ---------==//
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

#include "Inputs/lsc_usm_store.hpp"

constexpr uint32_t seed = 199;

template <int TestCastNum, typename T> bool tests() {
  bool passed = true;
  passed &= test<TestCastNum, T, 1, 4, 1, 32, true>();
  passed &= test<TestCastNum + 1, T, 2, 2, 1, 16, true>();
  passed &= test<TestCastNum + 2, T, 4, 4, 1, 4, true>();

  return passed;
}

int main(void) {
  srand(seed);
  bool passed = true;

  passed &= tests<0, uint8_t>();
  passed &= tests<3, uint16_t>();

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
