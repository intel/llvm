//==-- atomic_update_usm_dg2_pvc.cpp - DPC++ ESIMD on-device test----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------===//

// REQUIRES: gpu-intel-pvc || gpu-intel-dg2

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "Inputs/atomic_update.hpp"

int main(void) {
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;

  constexpr bool TestCacheHintProperties = true;
  passed &= test_main<TestCacheHintProperties>(q);

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
