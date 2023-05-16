//==------- ctor_move_fp_extra.cpp  - DPC++ ESIMD on-device test -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu, level_zero
// XREQUIRES: gpu
// TODO Remove the level_zero restriction once the test is supported on other
// platforms
// UNSUPPORTED: cuda, hip
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out
// XFAIL: *
// TODO Remove XFAIL once the simd vector provides move constructor
//
// Test for esimd move constructor
// The test creates source simd instance with reference data and invokes move
// constructor in different C++ contexts to create a new simd instance from the
// source simd instance. It is expected for a new simd instance to store
// bitwise same data as the one passed as the source simd constructor.

// The test proxy is used to verify the move constructor was called actually.
#define __ESIMD_ENABLE_TEST_PROXY

#include "ctor_move.hpp"

using namespace esimd_test::api::functional;

int main(int, char **) {
  bool passed = true;
  const auto types = get_tested_types<tested_types::fp_extra>();
  const auto sizes = get_all_sizes();
  const auto contexts =
      unnamed_type_pack<ctors::initializer, ctors::var_decl,
                        ctors::rval_in_expr, ctors::const_ref>::generate();

  sycl::queue queue(esimd_test::ESIMDSelector,
                    esimd_test::createExceptionHandler());

  // Run test for all combinations possible
  passed &=
      for_all_combinations<ctors::run_test>(types, sizes, contexts, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
