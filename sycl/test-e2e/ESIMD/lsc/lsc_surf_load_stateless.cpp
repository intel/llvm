//==------- lsc_surf_load_stateless.cpp - DPC++ ESIMD on-device test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -fsycl-esimd-force-stateless-mem %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// The test checks functionality of the lsc_block_load, lsc_prefetch, lsc_gather
// accessor-based ESIMD intrinsics when stateless memory accesses are enforced,
// i.e. accessor based accesses are automatically converted to stateless
// accesses.

#include "Inputs/lsc_surf_load.hpp"

constexpr uint32_t seed = 196;
constexpr lsc_data_size DS_U8U32 = lsc_data_size::u8u32;
constexpr lsc_data_size DS_U16U32 = lsc_data_size::u16u32;

template <int TestCastNum, typename T, lsc_data_size DS> bool tests() {
  bool passed = true;
  passed &= test<TestCastNum, T, 1, 1, 1, 1, false, DS>(rand());
  passed &= test<TestCastNum + 1, T, 2, 4, 16, 1, false, DS>(rand());
  passed &= test<TestCastNum + 2, T, 4, 16, 2, 1, false, DS>(rand());

  if constexpr (DS == lsc_data_size::default_size) {
    passed &= test<TestCastNum + 3, T, 1, 4, 1, 32, true, DS>();
    passed &= test<TestCastNum + 4, T, 2, 2, 1, 16, true, DS>();
    passed &= test<TestCastNum + 5, T, 4, 4, 1, 4, true, DS>();
  }

  return passed;
}

int main(void) {
  srand(seed);

  bool passed = true;
  passed &= tests<0, uint32_t, lsc_data_size::u8u32>();
  passed &= tests<3, uint32_t, lsc_data_size::u16u32>();
  passed &= tests<6, uint32_t, lsc_data_size::default_size>();
  passed &= tests<12, uint64_t, lsc_data_size::default_size>();

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
