//==------------ lsc_usm_block_load_u64.cpp - DPC++ ESIMD on-device test ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-dg2
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "Inputs/lsc_usm_block_load_prefetch.hpp"

constexpr uint32_t Seed = 187;

int main(void) {
  srand(Seed);
  bool Passed = true;
  Passed &= test_lsc_block_load<uint64_t>();
#ifdef USE_PVC
  Passed &= test_lsc_block_load<double>();
#endif

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
