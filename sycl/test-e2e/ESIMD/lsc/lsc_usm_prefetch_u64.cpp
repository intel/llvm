//==------------ lsc_usm_prefetch_u64.cpp - DPC++ ESIMD on-device test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "Inputs/lsc_usm_block_load_prefetch.hpp"
#include "Inputs/lsc_usm_gather_prefetch.hpp"

template <typename T> bool tests() {
  constexpr lsc_data_size DS = lsc_data_size::u64;
  constexpr bool GatherLikePrefetch = true;

  bool Passed = true;
  Passed &= test_lsc_prefetch<T, DS, GatherLikePrefetch>();
  Passed &= test_lsc_prefetch<T, DS, !GatherLikePrefetch>();

  return Passed;
}

int main(void) {
  constexpr uint32_t Seed = 188;
  srand(Seed);
  bool Passed = true;

  Passed &= tests<uint64_t>();
  Passed &= tests<double>();

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
