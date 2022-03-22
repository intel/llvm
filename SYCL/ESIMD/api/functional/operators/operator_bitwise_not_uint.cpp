//==------- operator_bitwise_not_uint.cpp  - DPC++ ESIMD on-device test ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu, level_zero
// XREQUIRES: gpu
// TODO gpu and level_zero in REQUIRES due to only this platforms supported yet.
// The current "REQUIRES" should be replaced with "gpu" only as mentioned in
// "XREQUIRES".
// UNSUPPORTED: cuda, hip
// RUN: %clangxx -fsycl %s -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// Test for simd bitwise not operator.
// The test creates source simd instance with reference data and invokes bitwise
// not operator.
// The test verifies that data from simd is not corrupted after calling bitwise
// not operator, that bitwise not operator return type is as expected and
// bitwise not operator result values are correct.

#include "operator_bitwise_not.hpp"

using namespace esimd_test::api::functional;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto uint_types = get_tested_types<tested_types::uint>();
  const auto all_dims = get_all_dimensions();

  // Running test for all uint types
  passed &= for_all_combinations<operators::run_test,
                                 operators::bitwise_not_operator>(
      uint_types, all_dims, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
