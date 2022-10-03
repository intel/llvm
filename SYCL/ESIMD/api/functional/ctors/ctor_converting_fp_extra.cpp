//==------- ctor_converting_fp_extra.cpp  - DPC++ ESIMD on-device test -----==//
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
// Test for simd converting constructor for extra fp types.
// This test uses extra fp data types with different dimensionality, base and
// step values and different simd constructor invocation contexts.
// The test do the following actions:
//  - construct simd with source data type, then construct simd with destination
//    type from the earlier constructed simd
//  - compare retrieved and expected values

#include "ctor_converting.hpp"

using namespace sycl::ext::intel::experimental::esimd;
using namespace esimd_test::api::functional;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector,
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto fp_types = get_tested_types<tested_types::fp_extra>();
  const auto uint_types = get_tested_types<tested_types::uint>();
  const auto sint_types = get_tested_types<tested_types::sint>();
  const auto core_types = get_tested_types<tested_types::core>();
  const auto single_size = get_sizes<1, 8>();
  const auto contexts =
      unnamed_type_pack<ctors::initializer, ctors::var_decl,
                        ctors::rval_in_expr, ctors::const_ref>::generate();

  // Run for specific combinations of types, vector length, base and step values
  // and invocation contexts.
  // The first types is the source types. the second types is the destination
  // types.
  passed &= for_all_combinations<ctors::run_test>(fp_types, single_size,
                                                  fp_types, contexts, queue);
  passed &= for_all_combinations<ctors::run_test>(fp_types, single_size,
                                                  uint_types, contexts, queue);
  passed &= for_all_combinations<ctors::run_test>(fp_types, single_size,
                                                  sint_types, contexts, queue);
  passed &= for_all_combinations<ctors::run_test>(uint_types, single_size,
                                                  core_types, contexts, queue);
  passed &= for_all_combinations<ctors::run_test>(sint_types, single_size,
                                                  uint_types, contexts, queue);
  passed &= for_all_combinations<ctors::run_test>(sint_types, single_size,
                                                  sint_types, contexts, queue);
  passed &= for_all_combinations<ctors::run_test>(sint_types, single_size,
                                                  fp_types, contexts, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
