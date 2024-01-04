//==------- atomic_update_acc.cpp - DPC++ ESIMD on-device test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %{build} -fsycl-esimd-force-stateless-mem -o %t.out
// RUN: %{run} %t.out

#include "Inputs/atomic_update.hpp"

int main(void) {
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  esimd_test::printTestLabel(q);

  constexpr bool TestCacheHintProperties = true;
  bool passed = test_main_acc<!TestCacheHintProperties>(q);

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
