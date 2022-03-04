//===-- ctor_copy.hpp - Functions for tests on simd copy constructor definition.
//      -------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functions for tests on simd copy constructor.
///
//===----------------------------------------------------------------------===//

#pragma once
#define ESIMD_TESTS_DISABLE_DEPRECATED_TEST_DESCRIPTION_FOR_LOGS

#include "common.hpp"

namespace esimd = sycl::ext::intel::experimental::esimd;

namespace esimd_test::api::functional::ctors {

// Descriptor class for the case of calling constructor in initializer context
struct initializer {
  static std::string get_description() { return "initializer"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT *const output_data) {
    const auto simd_by_init = esimd::simd<DataT, NumElems>();
    simd_by_init.copy_to(output_data);
  }
};

// Descriptor class for the case of calling constructor in variable declaration
// context
struct var_decl {
  static std::string get_description() { return "variable declaration"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT *const output_data) {
    esimd::simd<DataT, NumElems> simd_by_var_decl;
    simd_by_var_decl.copy_to(output_data);
  }
};

// Descriptor class for the case of calling constructor in rvalue in an
// expression context
struct rval_in_expr {
  static std::string get_description() { return "rvalue in an expression"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT *const output_data) {
    esimd::simd<DataT, NumElems> simd_by_rval;
    simd_by_rval = esimd::simd<DataT, NumElems>();
    simd_by_rval.copy_to(output_data);
  }
};

// Descriptor class for the case of calling constructor in const reference
// context
struct const_ref {
  static std::string get_description() { return "const reference"; }

  template <typename DataT, int NumElems>
  static void
  call_simd_by_const_ref(const esimd::simd<DataT, NumElems> &simd_by_const_ref,
                         DataT *output_data) {
    simd_by_const_ref.copy_to(output_data);
  }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT *const output_data) {
    call_simd_by_const_ref<DataT, NumElems>(esimd::simd<DataT, NumElems>(),
                                            output_data);
  }
};

// Struct that calls simd in provided context and then verifies obtained result.
template <typename DataT, typename SizeT, typename TestCaseT> struct run_test {
  static constexpr int NumElems = SizeT::value;
  using TestDescriptionT = ctors::TestDescription<NumElems, TestCaseT>;

  bool operator()(sycl::queue &queue, const std::string &data_type) {
    bool passed = true;
    log::trace<TestDescriptionT>(data_type);

    if (should_skip_test_with<DataT>(queue.get_device())) {
      return true;
    }

    // We use it to avoid empty functions being optimized out by compiler
    // checking the result of the simd calling because values of the constructed
    // object's elements are undefined.
    shared_vector<DataT> result(NumElems, shared_allocator<DataT>(queue));
    // We do not re-throw an exception to test all combinations of types and
    // vector sizes.
    try {
      queue.submit([&](sycl::handler &cgh) {
        DataT *const out = result.data();
        cgh.single_task<Kernel<DataT, NumElems, TestCaseT>>(
            [=]() SYCL_ESIMD_KERNEL {
              TestCaseT::template call_simd_ctor<DataT, NumElems>(out);
            });
      });
      queue.wait_and_throw();
    } catch (const sycl::exception &e) {
      passed = false;
      log::fail(TestDescriptionT(data_type), "A SYCL exception was caught: ",
                e.what());
    }

    return passed;
  }
};

} // namespace esimd_test::api::functional::ctors
