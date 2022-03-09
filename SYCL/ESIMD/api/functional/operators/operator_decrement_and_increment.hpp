//===-- operator_decrement_and_increment.hpp - Functions for tests on simd
//      increment and decrement operators definition. ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functions for tests on simd decrement and increment
/// operators.
///
//===----------------------------------------------------------------------===//

#pragma once
#define ESIMD_TESTS_DISABLE_DEPRECATED_TEST_DESCRIPTION_FOR_LOGS

#include "../mutator.hpp"
#include "common.hpp"

namespace esimd = sycl::ext::intel::esimd;
namespace esimd_functional = esimd_test::api::functional;

namespace esimd_test::api::functional::operators {

// Descriptor class for the case of calling constructor in initializer context.
struct pre_decrement {
  static std::string get_description() { return "pre decrement"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(const DataT *const ref_data,
                             DataT *const source_simd_out,
                             DataT *const result_simd_out) {
    auto source_simd = esimd::simd<DataT, NumElems>();
    source_simd.copy_from(ref_data);
    esimd::simd<DataT, NumElems> result_simd = --source_simd;
    source_simd.copy_to(source_simd_out);
    result_simd.copy_to(result_simd_out);
  }

  template <typename DataT> static DataT apply_operator(DataT &val) {
    return --val;
  }

  static constexpr bool is_increment() { return false; }
};

// Descriptor class for the case of calling constructor in initializer context.
struct post_decrement {
  static std::string get_description() { return "post decrement"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(const DataT *const ref_data,
                             DataT *const source_simd_out,
                             DataT *const result_simd_out) {
    auto source_simd = esimd::simd<DataT, NumElems>();
    source_simd.copy_from(ref_data);
    esimd::simd<DataT, NumElems> result_simd = source_simd--;
    source_simd.copy_to(source_simd_out);
    result_simd.copy_to(result_simd_out);
  }

  template <typename DataT> static DataT apply_operator(DataT &val) {
    return val--;
  }

  static constexpr bool is_increment() { return false; }
};

// Descriptor class for the case of calling constructor in initializer context.
struct pre_increment {
  static std::string get_description() { return "pre increment"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(const DataT *const ref_data,
                             DataT *const source_simd_out,
                             DataT *const result_simd_out) {
    auto source_simd = esimd::simd<DataT, NumElems>();
    source_simd.copy_from(ref_data);
    esimd::simd<DataT, NumElems> result_simd = ++source_simd;
    source_simd.copy_to(source_simd_out);
    result_simd.copy_to(result_simd_out);
  }

  template <typename DataT> static DataT apply_operator(DataT &val) {
    return ++val;
  }

  static constexpr bool is_increment() { return true; }
};

// Descriptor class for the case of calling constructor in initializer context.
struct post_increment {
  static std::string get_description() { return "post increment"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(const DataT *const ref_data,
                             DataT *const source_simd_out,
                             DataT *const result_simd_out) {
    auto source_simd = esimd::simd<DataT, NumElems>();
    source_simd.copy_from(ref_data);
    esimd::simd<DataT, NumElems> result_simd = source_simd++;
    source_simd.copy_to(source_simd_out);
    result_simd.copy_to(result_simd_out);
  }

  template <typename DataT> static DataT apply_operator(DataT &val) {
    return val++;
  }

  static constexpr bool is_increment() { return true; }
};

struct base_test {
  template <typename DataT, int NumElems, typename TestCaseT>
  static std::vector<DataT> generate_input_data() {
    std::vector<DataT> ref_data = generate_ref_data<DataT, NumElems>();

    if constexpr (TestCaseT::is_increment()) {
      mutate(ref_data, mutator::For_addition<DataT>(1));
    } else {
      mutate(ref_data, mutator::For_subtraction<DataT>(1));
    }

    return ref_data;
  }
};

struct fp_accuracy_test {
  template <typename DataT, int NumElems, typename TestCaseT>
  static std::vector<DataT> generate_input_data() {
    std::vector<DataT> ref_data;

    static const DataT min = value<DataT>::lowest();
    static const DataT denorm_min = value<DataT>::denorm_min();
    static const DataT max = value<DataT>::max();
    static const DataT inexact = static_cast<DataT>(0.1);

    if constexpr (TestCaseT::is_increment()) {
      ref_data.reserve((NumElems > 1) ? NumElems : 6);
      ref_data.insert(ref_data.end(),
                      {inexact, denorm_min, -denorm_min,
                       value<DataT>::pos_ulp(static_cast<DataT>(-1.0)),
                       value<DataT>::pos_ulp(min),
                       value<DataT>::neg_ulp(max - 1)});

    } else {
      ref_data.reserve((NumElems > 1) ? NumElems : 6);
      ref_data.insert(ref_data.end(),
                      {inexact, denorm_min, -denorm_min,
                       value<DataT>::neg_ulp(static_cast<DataT>(-1.0)),
                       value<DataT>::neg_ulp(max),
                       value<DataT>::pos_ulp(min + 1)});
    }

    for (size_t i = ref_data.size(); i < NumElems; ++i) {
      ref_data.push_back(inexact * i);
    }

    return ref_data;
  }
};

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename IsAccuracyTestT, typename DataT, typename SizeT,
          typename TestCaseT>
class run_test {
  static constexpr int NumElems = SizeT::value;
  using TestDescriptionT = operators::TestDescription<NumElems, TestCaseT>;

public:
  bool operator()(sycl::queue &queue, const std::string &data_type) {
    bool passed = true;
    std::vector<DataT> ref_data =
        IsAccuracyTestT::template generate_input_data<DataT, NumElems,
                                                      TestCaseT>();

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
    log::trace<TestDescriptionT>(data_type);

    shared_allocator<DataT> allocator(queue);
    shared_vector<DataT> source_simd_out(NumElems, allocator);
    shared_vector<DataT> result_simd_out(NumElems, allocator);
    shared_vector<DataT> shared_ref_data(ref_data.begin(), ref_data.end(),
                                         allocator);

    queue.submit([&](sycl::handler &cgh) {
      const DataT *const ref = shared_ref_data.data();
      DataT *source_simd_data_ptr = source_simd_out.data();
      DataT *result_simd_data_ptr = result_simd_out.data();

      cgh.single_task<Kernel<DataT, NumElems, TestCaseT>>(
          [=]() SYCL_ESIMD_KERNEL {
            TestCaseT::template call_simd_ctor<DataT, NumElems>(
                ref, source_simd_data_ptr, result_simd_data_ptr);
          });
    });
    queue.wait_and_throw();

    for (size_t i = 0; i < NumElems; ++i) {
      DataT expected_source_value = shared_ref_data[i];

      const DataT expected_return_value =
          TestCaseT::apply_operator(expected_source_value);

      passed &= verify_result(i, expected_source_value, source_simd_out[i],
                              "argument modification", data_type);
      passed &= verify_result(i, expected_return_value, result_simd_out[i],
                              "return value", data_type);
    }

    return passed;
  }

  bool verify_result(size_t i, DataT expected, DataT retrieved,
                     const std::string &value_description,
                     const std::string &data_type) {
    bool passed = true;
    if constexpr (type_traits::is_sycl_floating_point_v<DataT>) {
      if (std::isnan(expected) && !std::isnan(retrieved)) {
        passed = false;
        log::fail(TestDescriptionT(data_type), "Unexpected ",
                  value_description, "at index ", i, ", retrieved: ", retrieved,
                  ", expected: any NaN value");
      }
    }
    if (!are_bitwise_equal(expected, retrieved)) {
      passed = false;
      log::fail(TestDescriptionT(data_type), "Unexpected ",
                value_description, "at index ", i, ", retrieved: ", retrieved,
                ", expected: ", expected);
    }
    return passed;
  }
};

} // namespace esimd_test::api::functional::operators
