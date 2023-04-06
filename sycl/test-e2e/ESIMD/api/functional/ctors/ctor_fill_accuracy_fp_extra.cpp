//==------- ctor_fill_accuracy_fp_extra.cpp  - DPC++ ESIMD on-device test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu, level_zero
// UNSUPPORTED: gpu-intel-gen9 && windows
// XREQUIRES: gpu
// TODO gpu and level_zero in REQUIRES due to only this platforms supported yet.
// The current "REQUIRES" should be replaced with "gpu" only as mentioned in
// "XREQUIRES".
// UNSUPPORTED: cuda, hip
// RUN: %clangxx -fsycl %s -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// TODO simd<float, 32> fills with unexpected values while base value is denorm
// and step is ulp. The SIMD_RUN_TEST_WITH_DENORM_INIT_VAL_AND_ULP_STEP macros
// must be enabled when it is resolved.
//
// The test verifies that simd fill constructor has no precision differences.
// The test do the following actions:
//  - call simd with predefined base and step values
//  - bitwise comparing that output[0] value is equal to base value and
//    output[i] is equal to output[i -1] + step_value

#include "ctor_fill.hpp"

using namespace esimd_test::api::functional;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector,
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto types = get_tested_types<tested_types::fp_extra>();
  const auto single_size = get_sizes<8>();
  const auto context = unnamed_type_pack<ctors::var_decl>::generate();
// Run for specific combinations of types, base and step values and vector
// length.
// The first init_val value it's a base value and the second init_val value
// it's a step value.
#ifdef SIMD_RUN_TEST_WITH_DENORM_INIT_VAL_AND_ULP_STEP
  {
    const auto base_values =
        ctors::get_init_values_pack<ctors::init_val::denorm>();
    const auto step_values =
        ctors::get_init_values_pack<ctors::init_val::ulp>();
    passed &= for_all_combinations<ctors::run_test>(
        types, single_size, context, base_values, step_values, queue);
  }
#endif
  {
    const auto base_values =
        ctors::get_init_values_pack<ctors::init_val::inexact,
                                    ctors::init_val::min>();
    const auto step_values =
        ctors::get_init_values_pack<ctors::init_val::ulp,
                                    ctors::init_val::ulp_half>();
    passed &= for_all_combinations<ctors::run_test>(
        types, single_size, context, base_values, step_values, queue);
  }

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
