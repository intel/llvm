//==------------ lsc_usm_prefetch_u64.cpp - DPC++ ESIMD on-device test -----==//
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
#include "Inputs/lsc_usm_gather_prefetch.hpp"

template <int TestCastNum, typename T> bool tests() {
  constexpr lsc_data_size DS = lsc_data_size::u64;
  constexpr cache_hint L1H = cache_hint::cached;
  constexpr cache_hint L3H = cache_hint::uncached;
  constexpr bool DoPrefetch = true;

  bool Passed = true;
  // non transpose
#ifndef USE_SCALAR_OFFSET
  Passed &= test_lsc_prefetch<T, DS>();
#endif // !USE_SCALAR_OFFSET

  // transpose
  Passed &= test_lsc_prefetch<TestCastNum, T, DS>();

  return Passed;
}

int main(void) {
  constexpr uint32_t Seed = 188;
  srand(Seed);
  bool Passed = true;

  Passed &= tests<0, uint64_t>();
  Passed &= tests<10, double>();

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
