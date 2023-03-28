//==------- ctor_load_usm_core.cpp  - DPC++ ESIMD on-device test -----------==//
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
// Test for simd load constructor.
// The test uses reference data and different alignment flags. Invokes simd
// constructors in different contexts with provided reference data and alignment
// flag using USM pointer as input.
// It is expected for destination simd instance to store a bitwise same data as
// the reference one.

#include "ctor_load_usm.hpp"

using namespace esimd_test::api::functional;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector,
                    esimd_test::createExceptionHandler());

  bool passed = true;
  constexpr auto oword = 16;

  const auto types = get_tested_types<tested_types::core>();
  const auto sizes = get_all_sizes();

  const auto contexts =
      unnamed_type_pack<ctors::initializer, ctors::var_decl,
                        ctors::rval_in_expr, ctors::const_ref>::generate();
  const auto alignments =
      named_type_pack<ctors::alignment::element, ctors::alignment::vector,
                      ctors::alignment::overal<oword>>::generate();

  passed &= for_all_combinations<ctors::run_test>(types, sizes, contexts,
                                                  alignments, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
