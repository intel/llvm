//==------- ctor_vector_fp_extra.cpp  - DPC++ ESIMD on-device test ---------==//
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
// XRUN: %clangxx -fsycl %s -fsycl-device-code-split=per_kernel -o %t.out
// XRUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: false
// XFAIL: *
// TODO The simd can't be constructed with sycl::half data type. The issue was
// created (https://github.com/intel/llvm/issues/5077) and the test must be
// enabled when it is resolved.
//
// Test for simd constructor from vector.
// This test uses extra fp data types, dimensionality and different simd
// constructor invocation contexts.
// The test do the following actions:
//  - call init_simd.data() to retreive vector_type and then provide it to the
//    simd constructor
//  - bitwise comparing expected and retrieved values

#include "ctor_vector.hpp"

using namespace esimd_test::api::functional;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto types = get_tested_types<tested_types::fp_extra>();
  const auto dims = get_all_dimensions();
  const auto contexts =
      unnamed_type_pack<ctors::initializer, ctors::var_decl,
                        ctors::rval_in_expr, ctors::const_ref>::generate();

  // Run test for all combinations possible
  passed &= for_all_combinations<ctors::run_test>(types, dims, contexts, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
