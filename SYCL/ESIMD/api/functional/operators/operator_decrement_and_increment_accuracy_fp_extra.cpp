//==------- operator_decrement_and_increment_accuracy_fp_extra.cpp  - DPC++
//          ESIMD on-device test -------------------------------------------==//
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
// TODO simd<sycl::half, N> vector filled with unexpected values. The
// ESIMD_TESTS_RUN_WITH_HALF macros must be enabled when it is resolved.
//
// Test for simd increment and decrement operators.
// The test creates source simd instance and call increment or decrement
// operator opera with reference data. The test verifies that in the output data
// contained correctness data according to chosen operator and has no precision
// differences with interaction with floating point data types.

#include "operator_decrement_and_increment.hpp"

using namespace esimd_test::api::functional;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

#ifdef ESIMD_TESTS_RUN_WITH_HALF
  const auto fp_extra_types = get_tested_types<tested_types::fp_extra>();
#else
  const auto fp_extra_types = named_type_pack<double>::generate("double");
#endif
  const auto all_sizes = get_all_sizes();
  const auto contexts =
      unnamed_type_pack<operators::pre_increment, operators::post_increment,
                        operators::pre_decrement,
                        operators::post_decrement>::generate();

  passed &=
      for_all_combinations<operators::run_test, operators::fp_accuracy_test>(
          fp_extra_types, all_sizes, contexts, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
