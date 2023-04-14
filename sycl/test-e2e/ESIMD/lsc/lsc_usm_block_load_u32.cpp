//==------- lsc_usm_block_load_u32.cpp - DPC++ ESIMD on-device test --------==//
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

constexpr uint32_t Seed = 188;

int main(void) {
  srand(Seed);
  bool Passed = true;

  Passed &= test_lsc_block_load<uint32_t>();
  Passed &= test_lsc_block_load<float>();
  Passed &=
      test_lsc_block_load<sycl::ext::intel::experimental::esimd::tfloat32>();

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
