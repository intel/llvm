//==------- ctor_load_usm_fp_extra.cpp  - DPC++ ESIMD on-device test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out
//
// Test for simd load constructor.
// The test uses reference data and different alignment flags. Invokes simd
// constructors in different contexts with provided reference data and alignment
// flag using USM pointer as input.
// It is expected for destination simd instance to store a bitwise same data as
// the reference one.

// https://github.com/intel/llvm/issues/14650
// UNSUPPORTED: linux

#include "ctor_load_usm.hpp"

using namespace esimd_test::api::functional;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector,
                    esimd_test::createExceptionHandler());

  bool passed = true;
  constexpr auto oword = 16;

  const auto types = get_tested_types<tested_types::fp_extra>();
  const auto dims = get_all_dimensions();

  const auto contexts =
      unnamed_type_pack<ctors::initializer, ctors::var_decl,
                        ctors::rval_in_expr, ctors::const_ref>::generate();
  const auto alignments =
      named_type_pack<ctors::alignment::element, ctors::alignment::vector,
                      ctors::alignment::overal<oword>>::generate();

  passed &= for_all_combinations<ctors::run_test>(types, dims, contexts,
                                                  alignments, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
