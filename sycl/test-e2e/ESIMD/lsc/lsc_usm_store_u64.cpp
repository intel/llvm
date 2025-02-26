//==------- lsc_usm_store_u64.cpp - DPC++ ESIMD on-device test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc || gpu-intel-dg2
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "Inputs/lsc_usm_store.hpp"

constexpr uint32_t seed = 287;

template <int TestCastNum, typename T> bool tests() {
  bool passed = true;
  // non transpose
  passed &= test<TestCastNum, T, 1, 4, 32, 1, false>(rand());
  passed &= test<TestCastNum + 1, T, 1, 4, 32, 2, false>(rand());
  passed &= test<TestCastNum + 2, T, 1, 4, 16, 2, false>(rand());
  passed &= test<TestCastNum + 3, T, 1, 4, 4, 1, false>(rand());
  passed &= test<TestCastNum + 4, T, 1, 1, 1, 1, false>(1);
  passed &= test<TestCastNum + 5, T, 2, 1, 1, 1, false>(1);

  // passed &= test<TestCastNum + 6, T, 1, 4, 8, 2, false>(rand());
  // passed &= test<TestCastNum + 7, T, 1, 4, 8, 3, false>(rand());

  // transpose
  passed &= test<TestCastNum + 8, T, 1, 4, 1, 32, true>();
  passed &= test<TestCastNum + 9, T, 2, 2, 1, 16, true>();
  passed &= test<TestCastNum + 10, T, 4, 4, 1, 4, true>();
  return passed;
}

int main(void) {
  srand(seed);
  bool passed = true;
  auto Q = queue{gpu_selector_v};

  passed &= tests<0, uint64_t>();
  if (Q.get_device().has(sycl::aspect::fp64)) {
    passed &= tests<11, double>();
  }

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
