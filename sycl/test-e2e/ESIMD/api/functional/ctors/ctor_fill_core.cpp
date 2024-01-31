//==------- ctor_fill_core.cpp  - DPC++ ESIMD on-device test ---------------==//
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
// Test for simd fill constructor for core types.
// This test uses different data types, sizes, base and step values and
// different simd constructor invocation contexts. The test do the following
// actions:
//  - construct simd with pre-defined base and step value
//  - bitwise comparing expected and retrieved values

#include "ctor_fill.hpp"

using namespace esimd_test::api::functional;
using init_val = ctors::init_val;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector,
                    esimd_test::createExceptionHandler());

  bool passed = true;
  // Checks are run for specific combinations of types, vector length,
  // invocation contexts, base and step values accumulating the result in
  // boolean flag.

  {
    // Validate basic functionality works for every invocation context
    const auto types = named_type_pack<char, int>::generate("char", "int");
    const auto sizes = get_sizes<1, 8>();
    const auto contexts =
        unnamed_type_pack<ctors::initializer, ctors::var_decl,
                          ctors::rval_in_expr, ctors::const_ref>::generate();
    {
      const auto base_values =
          ctors::get_init_values_pack<init_val::min_half>();
      const auto step_values = ctors::get_init_values_pack<init_val::zero>();
      passed &= for_all_combinations<ctors::run_test>(
          types, sizes, contexts, base_values, step_values, queue);
    }
    {
      const auto base_values =
          ctors::get_init_values_pack<init_val::min_half, init_val::zero>();
      const auto step_values =
          ctors::get_init_values_pack<init_val::positive>();
      passed &= for_all_combinations<ctors::run_test>(
          types, sizes, contexts, base_values, step_values, queue);
    }
  }
  {
    // Validate basic functionality works for every type

    const auto types = get_tested_types<tested_types::core>();
    const auto sizes = get_all_sizes();
    const auto contexts = unnamed_type_pack<ctors::var_decl>::generate();
    {
      const auto base_values =
          ctors::get_init_values_pack<init_val::min, init_val::max_half>();
      const auto step_values = ctors::get_init_values_pack<init_val::zero>();
      passed &= for_all_combinations<ctors::run_test>(
          types, sizes, contexts, base_values, step_values, queue);
    }
    {
      const auto base_values =
          ctors::get_init_values_pack<init_val::zero, init_val::max_half>();
      const auto step_values =
          ctors::get_init_values_pack<init_val::positive, init_val::negative>();
      passed &= for_all_combinations<ctors::run_test>(
          types, sizes, contexts, base_values, step_values, queue);
    }
  }
  {
    // Verify specific cases for different type groups
    const auto sizes = get_sizes<8>();
    const auto contexts = unnamed_type_pack<ctors::var_decl>::generate();
    {
      const auto types = get_tested_types<tested_types::uint>();
      {
        const auto base_values =
            ctors::get_init_values_pack<init_val::min, init_val::max>();
        const auto step_values =
            ctors::get_init_values_pack<init_val::positive,
                                        init_val::negative>();
        passed &= for_all_combinations<ctors::run_test>(
            types, sizes, contexts, base_values, step_values, queue);
      }
    }
    {
      const auto types = get_tested_types<tested_types::sint>();
      {
        const auto base_values = ctors::get_init_values_pack<init_val::min>();
        const auto step_values =
            ctors::get_init_values_pack<init_val::positive>();
        passed &= for_all_combinations<ctors::run_test>(
            types, sizes, contexts, base_values, step_values, queue);
      }
      {
        const auto base_values = ctors::get_init_values_pack<init_val::max>();
        const auto step_values =
            ctors::get_init_values_pack<init_val::negative>();
        passed &= for_all_combinations<ctors::run_test>(
            types, sizes, contexts, base_values, step_values, queue);
      }
    }
    {
      const auto types = get_tested_types<tested_types::fp>();
      {
        const auto base_values =
            ctors::get_init_values_pack<init_val::negative>();
        const auto step_values =
            ctors::get_init_values_pack<init_val::positive>();
        passed &= for_all_combinations<ctors::run_test>(
            types, sizes, contexts, base_values, step_values, queue);
      }
      // The test cases below have never been guaranteed to work some certain
      // way with base and step values set to inf or non. They may or may not
      // work as expected by the checks in this test.
      {
        const auto base_values =
            ctors::get_init_values_pack<init_val::neg_inf>();
        const auto step_values =
            ctors::get_init_values_pack<init_val::positive>();
        passed &= for_all_combinations<ctors::run_test>(
            types, sizes, contexts, base_values, step_values, queue);
      }
      {
        const auto base_values = ctors::get_init_values_pack<init_val::max>();
        const auto step_values =
            ctors::get_init_values_pack<init_val::neg_inf>();
        passed &= for_all_combinations<ctors::run_test>(
            types, sizes, contexts, base_values, step_values, queue);
      }
      {
        const auto base_values = ctors::get_init_values_pack<init_val::nan>();
        const auto step_values =
            ctors::get_init_values_pack<init_val::negative>();
        passed &= for_all_combinations<ctors::run_test>(
            types, sizes, contexts, base_values, step_values, queue);
      }
      {
        const auto base_values =
            ctors::get_init_values_pack<init_val::negative>();
        const auto step_values = ctors::get_init_values_pack<init_val::nan>();
        passed &= for_all_combinations<ctors::run_test>(
            types, sizes, contexts, base_values, step_values, queue);
      }
    }
  }

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
