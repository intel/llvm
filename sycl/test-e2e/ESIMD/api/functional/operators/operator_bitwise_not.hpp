//===-- operator_bitwise_not.hpp - Functions for tests on simd assignment
//      operators. --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functions for tests on simd assignment operators.
///
//===----------------------------------------------------------------------===//

#pragma once

#define ESIMD_TESTS_DISABLE_DEPRECATED_TEST_DESCRIPTION_FOR_LOGS

#include "../mutator.hpp"
#include "../shared_element.hpp"
#include "common.hpp"
// For std::abs
#include <cmath>

namespace esimd = sycl::ext::intel::esimd;

namespace esimd_test::api::functional::operators {

// Descriptor class for the case of calling bitwise not operator.
struct bitwise_not_operator {
  static std::string get_description() { return "bitwise not"; }

  template <typename DataT, int NumElems>
  static bool call_operator(const DataT *const ref_data,
                            DataT *const source_simd_result,
                            DataT *const operator_result) {
    auto simd_obj = esimd::simd<DataT, NumElems>();
    simd_obj.copy_from(ref_data);
    const auto bitwise_not_result = ~simd_obj;
    simd_obj.copy_to(source_simd_result);
    bitwise_not_result.copy_to(operator_result);
    return std::is_same_v<decltype(~simd_obj), esimd::simd<DataT, NumElems>>;
  }
};

// Replace specific reference values to lower once.
template <typename T> struct For_bitwise_not {
  For_bitwise_not() = default;

  void operator()(T &val) {
    static_assert(!type_traits::is_sycl_floating_point_v<T>,
                  "Invalid data type.");
    if constexpr (std::is_signed_v<T>) {
      // We could have UB for negative zero in different integral type
      // representations: two's complement, ones' complement and signed
      // magnitude.
      // See http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2218.htm
      // Note that there is no check for UB with padding bits here, as it would
      // effectively disable any possible check for signed integer bitwise
      // operations.
      static const T max = value<T>::max();
      static const T lowest = value<T>::lowest();
      if (std::abs(lowest + 1) == (max - 1)) {
        // C11 standard mentions that it's possible to have a `0b100...0` value
        // as a trap value for twos' complement representation. In such case the
        // condition above would trigger for twos' complement representation
        // also.
        if (val == max) {
          // Would result in trap representation for signed magnitude
          val -= 1;
        } else if (val == 0) {
          // Would result in trap representation for ones' complement
          val = 1;
        }
      }
    } //  signed integral types
  }
};

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename TestCaseT, typename DataT, typename DimT> class run_test {
  static constexpr int NumElems = DimT::value;
  using TestDescriptionT = TestDescription<NumElems, TestCaseT>;

public:
  bool operator()(sycl::queue &queue, const std::string &data_type) {
    bool passed = true;
    std::vector<DataT> ref_data = generate_ref_data<DataT, NumElems>();

    mutate(ref_data, For_bitwise_not<DataT>());

    // If current number of elements is equal to one, then run test with each
    // one value from reference data.
    // If current number of elements is greater than one, then run tests with
    // whole reference data.
    if constexpr (NumElems == 1) {
      for (size_t i = 0; i < ref_data.size(); ++i) {
        passed = run_verification(queue, {ref_data[i]}, data_type);
      }
    } else {
      passed = run_verification(queue, ref_data, data_type);
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
    shared_vector<DataT> shared_ref_data(ref_data.begin(), ref_data.end(),
                                         allocator);
    shared_vector<DataT> source_simd_result(NumElems, allocator);
    shared_vector<DataT> operator_result(NumElems, allocator);

    shared_element<bool> is_correct_type(queue, true);

    queue.submit([&](sycl::handler &cgh) {
      const DataT *const ref = shared_ref_data.data();
      DataT *const source_simd_result_data_ptr = source_simd_result.data();
      DataT *const operator_result_data_ptr = operator_result.data();
      auto is_correct_type_ptr = is_correct_type.data();

      cgh.single_task<Kernel<DataT, NumElems, TestCaseT>>(
          [=]() SYCL_ESIMD_KERNEL {
            *is_correct_type_ptr =
                TestCaseT::template call_operator<DataT, NumElems>(
                    ref, source_simd_result_data_ptr, operator_result_data_ptr);
          });
    });
    queue.wait_and_throw();

    for (size_t i = 0; i < NumElems; ++i) {
      {
        const DataT &retrieved = source_simd_result[i];
        const DataT &expected = ref_data[i];
        if (!are_bitwise_equal(expected, retrieved)) {
          passed = false;
          log::fail(TestDescriptionT(data_type),
                    "Unexpected source simd value at index ", i,
                    ", retrieved: ", retrieved, ", expected: ", expected);
        }
      }
      {
        const DataT &retrieved = operator_result[i];
        const DataT &expected = ~shared_ref_data[i];
        if (!are_bitwise_equal(expected, retrieved)) {
          passed = false;
          log::fail(TestDescriptionT(data_type),
                    "Unexpected result value at index ", i,
                    ", retrieved: ", retrieved, ", expected: ", expected);
        }
      }
    }

    if (!is_correct_type.value()) {
      passed = false;
      log::fail(TestDescriptionT(data_type),
                "Invalid return type for operator.");
    }

    return passed;
  }
};

} // namespace esimd_test::api::functional::operators
