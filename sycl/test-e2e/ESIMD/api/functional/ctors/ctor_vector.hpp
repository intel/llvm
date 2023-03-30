//===-- ctor_vector.hpp - Functions for tests on simd vector constructor
//      definition. -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functions for tests on simd vector constructor.
///
//===----------------------------------------------------------------------===//

#pragma once
#define ESIMD_TESTS_DISABLE_DEPRECATED_TEST_DESCRIPTION_FOR_LOGS

#include "common.hpp"

namespace esimd = sycl::ext::intel::esimd;

namespace esimd_test::api::functional::ctors {

// Descriptor class for the case of calling constructor in initializer context.
struct initializer {
  static std::string get_description() { return "initializer"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(const DataT *const ref_data, DataT *const out) {
    esimd::simd<DataT, NumElems> init_simd(ref_data);
    const auto simd_by_init = esimd::simd<DataT, NumElems>(init_simd.data());
    simd_by_init.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in variable declaration
// context.
struct var_decl {
  static std::string get_description() { return "variable declaration"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(const DataT *const ref_data, DataT *const out) {
    esimd::simd<DataT, NumElems> init_simd(ref_data);
    esimd::simd<DataT, NumElems> simd_by_var_decl(init_simd.data());
    simd_by_var_decl.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in rvalue in an
// expression context.
struct rval_in_expr {
  static std::string get_description() { return "rvalue in an expression"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(const DataT *const ref_data, DataT *const out) {
    esimd::simd<DataT, NumElems> init_simd(ref_data);
    esimd::simd<DataT, NumElems> simd_by_rval;
    simd_by_rval = esimd::simd<DataT, NumElems>(init_simd.data());
    simd_by_rval.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in const reference
// context.
class const_ref {
public:
  static std::string get_description() { return "const reference"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(const DataT *const ref_data, DataT *const out) {
    esimd::simd<DataT, NumElems> init_simd(ref_data);
    call_simd_by_const_ref<DataT, NumElems>(
        esimd::simd<DataT, NumElems>(init_simd.data()), out);
  }

private:
  template <typename DataT, int NumElems>
  static void
  call_simd_by_const_ref(const esimd::simd<DataT, NumElems> &simd_by_const_ref,
                         DataT *out) {
    simd_by_const_ref.copy_to(out);
  }
};

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename DataT, typename SizeT, typename TestCaseT> class run_test {
  static constexpr int NumElems = SizeT::value;
  using TestDescriptionT = ctors::TestDescription<NumElems, TestCaseT>;

public:
  bool operator()(sycl::queue &queue, const std::string &data_type) {
    bool passed = true;
    log::trace<TestDescriptionT>(data_type);

    if (should_skip_test_with<DataT>(queue.get_device())) {
      return true;
    }

    const std::vector<DataT> ref_data = generate_ref_data<DataT, NumElems>();

    // If current number of elements is equal to one, then run test with each
    // one value from reference data.
    // If current number of elements is greater than one, then run tests with
    // whole reference data.
    if constexpr (NumElems == 1) {
      for (size_t i = 0; i < ref_data.size(); ++i) {
        passed &= run_verification(queue, {ref_data[i]}, data_type);
      }
    } else {
      passed &= run_verification(queue, ref_data, data_type);
    }
    return passed;
  }

private:
  bool run_verification(sycl::queue &queue, const std::vector<DataT> &ref_data,
                        const std::string &data_type) {
    assert(ref_data.size() == NumElems &&
           "Reference data size is not equal to the simd vector length.");

    bool passed = true;

    shared_allocator<DataT> allocator(queue);
    shared_vector<DataT> result(NumElems, allocator);
    shared_vector<DataT> shared_ref_data(ref_data.begin(), ref_data.end(),
                                         allocator);

    queue.submit([&](sycl::handler &cgh) {
      const DataT *const ref = shared_ref_data.data();
      DataT *const out = result.data();

      cgh.single_task<Kernel<DataT, NumElems, TestCaseT>>(
          [=]() SYCL_ESIMD_KERNEL {
            TestCaseT::template call_simd_ctor<DataT, NumElems>(ref, out);
          });
    });
    queue.wait_and_throw();

    for (size_t i = 0; i < result.size(); ++i) {
      const auto &expected = ref_data[i];
      const auto &retrieved = result[i];

      if (!are_bitwise_equal(expected, retrieved)) {
        passed = false;
        log::fail(TestDescriptionT(data_type), "Unexpected value at index ", i,
                  ", retrieved: ", retrieved, ", expected: ", expected);
      }
    }

    return passed;
  }
};

} // namespace esimd_test::api::functional::ctors
