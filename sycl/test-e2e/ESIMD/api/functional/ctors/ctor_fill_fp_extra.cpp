//==------- ctor_fill_fp_extra.cpp  - DPC++ ESIMD on-device test -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// TODO: remove fno-fast-math option once the issue is investigated and the test
// is fixed.
// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}
// RUN: %{build} -fsycl-device-code-split=per_kernel %{mathflags} -o %t.out
// RUN: %{run} %t.out
//
// Test for simd fill constructor for extra fp types.
// This test uses extra fp data types with different sizes, base and step values
// and different simd constructor invocation contexts.
// The test do the following actions:
//  - construct simd with pre-defined base and step value
//  - bitwise comparing expected and retrieved values

#include "ctor_fill.hpp"

using namespace esimd_test::api::functional;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector,
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto fp_types = get_tested_types<tested_types::fp_extra>();
  const auto single_size = get_sizes<8>();
  const auto contexts = unnamed_type_pack<ctors::var_decl>::generate();

  passed &= for_all_combinations<ctors::run_test>(
      fp_types, single_size, contexts,
      ctors::get_init_values_pack<ctors::init_val::neg_inf>(),
      ctors::get_init_values_pack<ctors::init_val::zero>(), queue);
  passed &= for_all_combinations<ctors::run_test>(
      fp_types, single_size, contexts,
      ctors::get_init_values_pack<ctors::init_val::max>(),
      ctors::get_init_values_pack<ctors::init_val::neg_inf>(), queue);
  passed &= for_all_combinations<ctors::run_test>(
      fp_types, single_size, contexts,
      ctors::get_init_values_pack<ctors::init_val::nan>(),
      ctors::get_init_values_pack<ctors::init_val::negative>(), queue);
  passed &= for_all_combinations<ctors::run_test>(
      fp_types, single_size, contexts,
      ctors::get_init_values_pack<ctors::init_val::zero>(),
      ctors::get_init_values_pack<ctors::init_val::nan>(), queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
