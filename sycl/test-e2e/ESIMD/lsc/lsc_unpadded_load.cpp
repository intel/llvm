//==--------------- lsc_unpadded_load.cpp - DPC++ ESIMD on-device test ------==/
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===/
// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "Inputs/lsc_load_prefetch_2d.hpp"

constexpr uint32_t seed = 322;

int main(void) {
  srand(seed);
  bool passed = true;

  // These parameters require unpadding.
  passed &= test<0, uint16_t, 2, 2, 2, 2>(16, 4, 16, 0, 0);
  passed &= test<1, uint32_t, 2, 2, 3, 2>(16, 4, 16, 1, 1);
  passed &= test<2, uint64_t, 2, 2, 2, 2>(16, 4, 16, 1, 1);
  passed &= test<3, uint8_t, 2, 2, 4, 2, 4>(16, 4, 16, 0, 0);

  return passed ? 0 : 1;
}
