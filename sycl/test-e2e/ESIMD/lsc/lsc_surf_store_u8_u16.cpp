//==------- lsc_surf_store_u8_u16.cpp - DPC++ ESIMD on-device test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "Inputs/lsc_surf_store.hpp"

constexpr uint32_t seed = 199;

template <int TestCastNum, typename T> bool tests() {
  bool passed = true;
  passed &= test<TestCastNum, T, 1, 4, 1, 32, true>();
  passed &= test<TestCastNum + 1, T, 2, 2, 1, 16, true>();
  passed &= test<TestCastNum + 2, T, 4, 4, 1, 4, true>();
  passed &= test<TestCastNum + 3, T, 4, 4, 1, 128, true,
                 lsc_data_size::default_size, cache_hint::none,
                 cache_hint::none, __ESIMD_NS::overaligned_tag<8>>();
  passed &= test<TestCastNum + 4, T, 4, 4, 1, 256, true,
                 lsc_data_size::default_size, cache_hint::none,
                 cache_hint::none, __ESIMD_NS::overaligned_tag<8>>();

  return passed;
}

int main(void) {
  srand(seed);
  bool passed = true;

  passed &= tests<0, uint8_t>();
  passed &= tests<5, uint16_t>();

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
