//==------- ctor_copy.cpp  - DPC++ ESIMD on-device test --------------------==//
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
// TODO Unexpected static_assert was retrieved while calling simd::copy_from()
// function. The issue was created (https://github.com/intel/llvm/issues/5112)
// and the test must be enabled when it is resolved.
//
// Test for esimd copy constructor.

#include "common.hpp"

using namespace sycl;
using namespace sycl::ext::intel::experimental::esimd;
using namespace esimd_test::api::functional::ctors;
using namespace esimd_test::api::functional;

// Descriptor class for the case of calling constructor in initializer context.
struct initializer {
  static std::string get_description() { return "initializer"; }

  template <typename DataT, int NumElems>
  static simd<DataT, NumElems> call_simd_ctor(const DataT *ref_data) {
    simd<DataT, NumElems> source_simd;
    source_simd.copy_from(ref_data);
    simd<DataT, NumElems> simd_by_init = simd<DataT, NumElems>(source_simd);
    return simd_by_init;
  }
};

// Descriptor class for the case of calling constructor in variable declaration
// context.
struct var_decl {
  static std::string get_description() { return "variable declaration"; }

  template <typename DataT, int NumElems>
  static simd<DataT, NumElems> call_simd_ctor(const DataT *ref_data) {
    simd<DataT, NumElems> source_simd;
    source_simd.copy_from(ref_data);
    simd<DataT, NumElems> simd_by_var_decl{source_simd};
    return simd_by_var_decl;
  }
};

// Descriptor class for the case of calling constructor in rvalue in an
// expression context.
struct rval_in_expr {
  static std::string get_description() { return "rvalue in an expression"; }

  template <typename DataT, int NumElems>
  static simd<DataT, NumElems> call_simd_ctor(const DataT *ref_data) {
    simd<DataT, NumElems> source_simd;
    source_simd.copy_from(ref_data);
    simd<DataT, NumElems> simd_by_rval;
    simd_by_rval = simd<DataT, NumElems>(source_simd);
    return simd_by_rval;
  }
};

// Descriptor class for the case of calling constructor in const reference
// context.
class const_ref {
public:
  static std::string get_description() { return "const reference"; }

  template <typename DataT, int NumElems>
  static simd<DataT, NumElems> call_simd_ctor(const DataT *ref_data) {
    simd<DataT, NumElems> source_simd;
    source_simd.copy_from(ref_data);
    return call_simd_by_const_ref<DataT, NumElems>(
        simd<DataT, NumElems>(source_simd));
  }

private:
  template <typename DataT, int NumElems>
  static simd<DataT, NumElems>
  call_simd_by_const_ref(const simd<DataT, NumElems> &simd_by_const_ref) {
    return simd_by_const_ref;
  }
};

int main(int argc, char **argv) {
  sycl::queue queue{esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler()};

  bool passed{true};

  const auto types{get_tested_types<tested_types::all>()};
  const auto dims{get_all_dimensions()};

  passed &= for_all_types_and_dims<test, initializer>(types, dims, queue);
  passed &= for_all_types_and_dims<test, var_decl>(types, dims, queue);
  passed &= for_all_types_and_dims<test, rval_in_expr>(types, dims, queue);
  passed &= for_all_types_and_dims<test, const_ref>(types, dims, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
