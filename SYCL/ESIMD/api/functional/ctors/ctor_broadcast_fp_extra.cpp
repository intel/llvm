//==------- ctor_broadcast_fp_extra.cpp  - DPC++ ESIMD on-device test ------==//
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
// Test for simd broadcast constructor.
// This test uses fp extra data types, sizes and different simd constructor
// invocation contexts.
// Type of a value that will be provided to the broadcast constructor may be
// differ, than value, that will be provided to the simd when it will be
// constructed. It is expected for a new simd instance to store same data as the
// one passed as the source simd constructor.

#include "ctor_broadcast.hpp"

using namespace sycl::ext::intel::esimd;
using namespace esimd_test::api::functional;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector,
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto uint_types = get_tested_types<tested_types::uint>();
  const auto sint_types = get_tested_types<tested_types::sint>();
  const auto fp_extra_types = get_tested_types<tested_types::fp_extra>();
  const auto single_size = get_sizes<8>();
  const auto context = unnamed_type_pack<ctors::var_decl>::generate();

  // Run for specific combinations of types, vector length, base and step values
  // and invocation contexts.
  // The source types is the first types, that provided to the
  // "for_all_combinations" the destination types is the second types that
  // provided to the "for_all_combinations".
  passed &= for_all_combinations<ctors::run_test_with_all_values>(
      fp_extra_types, single_size, fp_extra_types, context, queue);
  passed &= for_all_combinations<ctors::run_test_with_all_values>(
      fp_extra_types, single_size, uint_types, context, queue);
  passed &= for_all_combinations<ctors::run_test_with_all_values>(
      fp_extra_types, single_size, sint_types, context, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
