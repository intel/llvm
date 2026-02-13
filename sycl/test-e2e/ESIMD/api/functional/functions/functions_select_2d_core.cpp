//==------- functions_select_2d_core.cpp  - DPC++ ESIMD on-device test
//          ----------------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out
//
// Test for simd select for 2d function.
// The test creates source simd instance with reference data and invokes logical
// not operator, using core data types.
// The test verifies that selected values can be changed with avoid to change
// values, that hasn't beed selected.

#include "functions_select_2d.hpp"

using namespace sycl::ext::intel::experimental::esimd;
using namespace esimd_test::api::functional;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector,
                    esimd_test::createExceptionHandler());

  bool passed = functions::run_test_for_types<tested_types::core>(queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
