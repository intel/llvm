//==------------ lsc_usm_block_load_u64.cpp - DPC++ ESIMD on-device test ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "Inputs/lsc_usm_block_load_prefetch.hpp"

constexpr uint32_t Seed = 187;
template <typename T> bool tests() {
  bool Passed = true;
  Passed &= test<T, 32>(1, 4);
  Passed &= test<T, 16>(2, 2);
  Passed &= test<T, 4>(4, 4);
  return Passed;
}

int main(void) {
  srand(Seed);
  bool Passed = true;

  Passed &= tests<uint64_t>();
  Passed &= tests<double>();

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
