//==------- ctor_fill_core.cpp  - DPC++ ESIMD on-device test ---------------==//
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
// TODO This test disabled due to simd<short, 32> vector filled with unexpected
// values from 16th element. The issue was created
// https://github.com/intel/llvm/issues/5245 and and the
// SIMD_RUN_TEST_WITH_VECTOR_LEN_32 macros must be enabled when it is resolved.
//
// Test for simd fill constructor for core types.
// This test uses different data types, dimensionality, base and step values and
// different simd constructor invocation contexts. The test do the following
// actions:
//  - construct simd with pre-defined base and step value
//  - bitwise comparing expected and retrieved values

#include "ctor_fill.hpp"

using namespace sycl::ext::intel::experimental::esimd;
using namespace esimd_test::api::functional;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto two_dims = values_pack<1, 8>();
  const auto char_int_types = named_type_pack<char, int>({"char", "int"});

  // Run for specific combinations of types, vector length, base and step values
  // and invocation contexts.
  // The first init_val value it's a base value and the second init_val value
  // it's a step value.
  passed &=
      ctors::run_verification<ctors::initializer, ctors::init_val::min_half,
                              ctors::init_val::zero>(queue, two_dims,
                                                     char_int_types);

  passed &= ctors::run_verification<ctors::initializer, ctors::init_val::zero,
                                    ctors::init_val::positive>(queue, two_dims,
                                                               char_int_types);
  passed &=
      ctors::run_verification<ctors::initializer, ctors::init_val::min_half,
                              ctors::init_val::positive>(queue, two_dims,
                                                         char_int_types);

  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::min_half,
                                    ctors::init_val::zero>(queue, two_dims,
                                                           char_int_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::zero,
                                    ctors::init_val::positive>(queue, two_dims,
                                                               char_int_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::min_half,
                                    ctors::init_val::positive>(queue, two_dims,
                                                               char_int_types);

  passed &=
      ctors::run_verification<ctors::rval_in_express, ctors::init_val::min_half,
                              ctors::init_val::zero>(queue, two_dims,
                                                     char_int_types);
  passed &=
      ctors::run_verification<ctors::rval_in_express, ctors::init_val::zero,
                              ctors::init_val::positive>(queue, two_dims,
                                                         char_int_types);
  passed &=
      ctors::run_verification<ctors::rval_in_express, ctors::init_val::min_half,
                              ctors::init_val::positive>(queue, two_dims,
                                                         char_int_types);

  passed &= ctors::run_verification<ctors::const_ref, ctors::init_val::min_half,
                                    ctors::init_val::zero>(queue, two_dims,
                                                           char_int_types);
  passed &= ctors::run_verification<ctors::const_ref, ctors::init_val::zero,
                                    ctors::init_val::positive>(queue, two_dims,
                                                               char_int_types);
  passed &= ctors::run_verification<ctors::const_ref, ctors::init_val::min_half,
                                    ctors::init_val::positive>(queue, two_dims,
                                                               char_int_types);
#ifdef SIMD_RUN_TEST_WITH_VECTOR_LEN_32
  const auto all_dims = values_pack<1, 8, 16, 32>();
#else
  const auto all_dims = values_pack<1, 8, 16>();
#endif
  const auto all_types = get_tested_types<tested_types::all>();
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::min,
                                    ctors::init_val::zero>(queue, all_dims,
                                                           all_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::max_half,
                                    ctors::init_val::zero>(queue, all_dims,
                                                           all_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::zero,
                                    ctors::init_val::positive>(queue, all_dims,
                                                               all_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::max_half,
                                    ctors::init_val::positive>(queue, all_dims,
                                                               all_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::zero,
                                    ctors::init_val::negative>(queue, all_dims,
                                                               all_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::max_half,
                                    ctors::init_val::negative>(queue, all_dims,
                                                               all_types);

  const auto single_dim = values_pack<8>();
  const auto uint_types = get_tested_types<tested_types::uint>();
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::min,
                                    ctors::init_val::positive>(
      queue, single_dim, uint_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::min,
                                    ctors::init_val::negative>(
      queue, single_dim, uint_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::max,
                                    ctors::init_val::positive>(
      queue, single_dim, uint_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::max,
                                    ctors::init_val::negative>(
      queue, single_dim, uint_types);

  const auto sint_types = get_tested_types<tested_types::sint>();
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::min,
                                    ctors::init_val::positive>(
      queue, single_dim, uint_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::max,
                                    ctors::init_val::negative>(
      queue, single_dim, uint_types);

  const auto fp_types = get_tested_types<tested_types::fp>();
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::neg_inf,
                                    ctors::init_val::max>(queue, single_dim,
                                                          fp_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::max,
                                    ctors::init_val::neg_inf>(queue, single_dim,
                                                              fp_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::nan,
                                    ctors::init_val::negative>(
      queue, single_dim, fp_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::negative,
                                    ctors::init_val::nan>(queue, single_dim,
                                                          fp_types);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
