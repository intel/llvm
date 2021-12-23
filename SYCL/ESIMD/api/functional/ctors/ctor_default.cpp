//==------- ctor_default.cpp  - DPC++ ESIMD on-device test ----------------==//
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
// Test for esimd default constructor.
//
// The simd can't be constructed with sycl::half data type. The issue was
// created (https://github.com/intel/llvm/issues/5077) and the TEST_HALF macros
// must be enabled when it is resolved.

#include "common.hpp"

using namespace sycl::ext::intel::experimental::esimd;
using namespace esimd_test::api::functional;

// Descriptor class for the case of calling constructor in initializer context
struct initializer {
  static std::string get_description() { return "initializer"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT *const output_data) {
    const auto simd_by_init = simd<DataT, NumElems>();
    simd_by_init.copy_to(output_data);
  }
};

// Descriptor class for the case of calling constructor in variable declaration
// context
struct var_decl {
  static std::string get_description() { return "variable declaration"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT *const output_data) {
    simd<DataT, NumElems> simd_by_var_decl;
    simd_by_var_decl.copy_to(output_data);
  }
};

// Descriptor class for the case of calling constructor in rvalue in an
// expression context
struct rval_in_expr {
  static std::string get_description() { return "rvalue in an expression"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT *const output_data) {
    simd<DataT, NumElems> simd_by_rval;
    simd_by_rval = simd<DataT, NumElems>();
    simd_by_rval.copy_to(output_data);
  }
};

// Descriptor class for the case of calling constructor in const reference
// context
struct const_ref {
  static std::string get_description() { return "const reference"; }

  template <typename DataT, int NumElems>
  static void
  call_simd_by_const_ref(const simd<DataT, NumElems> &simd_by_const_ref,
                         DataT *output_data) {
    simd_by_const_ref.copy_to(output_data);
  }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT *const output_data) {
    call_simd_by_const_ref<DataT, NumElems>(simd<DataT, NumElems>(),
                                            output_data);
  }
};

// Struct that calls simd in provided context and then verifies obtained result.
template <typename DataT, int NumElems, typename TestCaseT> struct run_test {
  bool operator()(sycl::queue &queue, const std::string &data_type) {
    bool passed = true;
    DataT default_val{};

    shared_vector<DataT> result(NumElems, shared_allocator<DataT>(queue));

    queue.submit([&](sycl::handler &cgh) {
      DataT *const out = result.data();
      cgh.single_task<ctors::Kernel<DataT, NumElems, TestCaseT>>(
          [=]() SYCL_ESIMD_KERNEL {
            TestCaseT::template call_simd_ctor<DataT, NumElems>(out);
          });
    });

    for (size_t i = 0; i < result.size(); ++i) {
      if (result[i] != default_val) {
        passed = false;

        const auto description =
            ctors::TestDescription<DataT, NumElems, TestCaseT>(
                i, result[i], default_val, data_type);
        log::fail(description);
      }
    }

    return passed;
  }
};

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto types = get_tested_types<tested_types::all>();
  const auto dims = get_all_dimensions();

  passed &= for_all_types_and_dims<run_test, initializer>(types, dims, queue);
  passed &= for_all_types_and_dims<run_test, var_decl>(types, dims, queue);
  passed &= for_all_types_and_dims<run_test, rval_in_expr>(types, dims, queue);
  passed &= for_all_types_and_dims<run_test, const_ref>(types, dims, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
