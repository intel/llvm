//==------- operator_decrement_and_increment_core.cpp  - DPC++ ESIMD on-device
//          test -----------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out
//
// Test for simd increment and decrement operators.
// The test creates source simd instance and call increment or decrement
// operator opera with reference data. The test verifies that in the output data
// contained correctness data according to chosen operator.

#include "operator_decrement_and_increment.hpp"

using namespace esimd_test::api::functional;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector,
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto core_types = get_tested_types<tested_types::core>();
  const auto all_sizes = get_all_sizes();
  const auto operators =
      unnamed_type_pack<operators::pre_increment, operators::post_increment,
                        operators::pre_decrement,
                        operators::post_decrement>::generate();

  passed &= for_all_combinations<operators::run_test, operators::base_test>(
      core_types, all_sizes, operators, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
