//==------- ctor_copy_fp_extra.cpp  - DPC++ ESIMD on-device test -----------==//
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
// Test for simd copy constructor.
// This test uses different data types, sizes and different simd constructor
// invocation contexts.
// The test do the following actions:
//  - construct simd object with golden values calls copy constructor from
//    constructed simd object
//  - bitwise comparing expected and retrieved values

#include "ctor_copy.hpp"

using namespace esimd_test::api::functional;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto types = get_tested_types<tested_types::fp_extra>();
  const auto sizes = get_all_sizes();
  const auto contexts =
      unnamed_type_pack<ctors::initializer, ctors::var_decl,
                        ctors::rval_in_expr, ctors::const_ref>::generate();

  passed &=
      for_all_combinations<ctors::run_test>(types, sizes, contexts, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
