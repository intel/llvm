//==------- operator_logical_not.cpp  - DPC++ ESIMD on-device test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out
//
// Test for simd logical not operator.
// The test creates source simd instance with reference data and invokes logical
// not operator.
// The test verifies that data from simd is not corrupted after calling logical
// not operator, that logical not operator return type is as expected and
// logical not operator result values is correct.

#include "../shared_element.hpp"
#include "common.hpp"

using namespace sycl::ext::intel::esimd;
using namespace esimd_test::api::functional;

// Descriptor class for the case of calling logical not operator.
struct logical_not_operator {
  static std::string get_description() { return "logical not"; }

  template <typename DataT, int NumElems, typename OperatorResultT>
  static bool call_operator(const DataT *const ref_data, DataT *const out,
                            OperatorResultT *const operator_result) {
    simd<DataT, NumElems> simd_obj;
    simd_obj.copy_from(ref_data);
    auto logical_not_result = !simd_obj;

    for (size_t i = 0; i < NumElems; ++i) {
      operator_result[i] = logical_not_result[i];
    }

    simd_obj.copy_to(out);
    return std::is_same_v<decltype(!simd_obj), simd_mask<NumElems>>;
  }
};

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename TestCaseT, typename DataT, typename DimT> class run_test {
  static constexpr int NumElems = DimT::value;

public:
  bool operator()(sycl::queue &queue, const std::string &data_type) {
    bool passed = true;
    const std::vector<DataT> ref_data = generate_ref_data<DataT, NumElems>();

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
    shared_vector<DataT> result(NumElems, allocator);
    shared_vector<DataT> shared_ref_data(ref_data.begin(), ref_data.end(),
                                         allocator);

    shared_vector<int> operator_result(NumElems, shared_allocator<int>(queue));
    shared_element<bool> is_correct_type(queue, true);

    queue.submit([&](sycl::handler &cgh) {
      const DataT *const ref = shared_ref_data.data();
      DataT *const out = result.data();
      auto operator_result_storage = operator_result.data();
      auto is_correct_type_storage = is_correct_type.data();

      cgh.single_task<Kernel<DataT, NumElems, TestCaseT>>(
          [=]() SYCL_ESIMD_KERNEL {
            *is_correct_type_storage =
                TestCaseT::template call_operator<DataT, NumElems>(
                    ref, out, operator_result_storage);
          });
    });
    queue.wait_and_throw();

    for (size_t i = 0; i < result.size(); ++i) {
      if (!are_bitwise_equal(ref_data[i], result[i])) {
        passed = false;

        const auto description = operators::TestDescription<DataT, NumElems>(
            i, result[i], ref_data[i], data_type);
        log::fail(description);
      }
      auto retrieved = operator_result[i];
      auto expected = shared_ref_data[i] == 0;
      if (expected != retrieved) {
        passed = false;
        const auto description = operators::TestDescription<DataT, NumElems>(
            i, retrieved, expected, data_type);
        log::fail(description);
      }
    }

    if (!is_correct_type.value()) {
      passed = false;
      log::note("Test failed due to type of the object that returns " +
                TestCaseT::get_description() +
                " operator is not equal to the expected one for simd<" +
                data_type + ", " + std::to_string(NumElems) + ">.");
    }

    return passed;
  }
};

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector,
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto uint_types = get_tested_types<tested_types::uint>();
  const auto sint_types = get_tested_types<tested_types::sint>();
  const auto all_dims = get_all_dimensions();

  passed &= for_all_combinations<run_test, logical_not_operator>(
      uint_types, all_dims, queue);
  passed &= for_all_combinations<run_test, logical_not_operator>(
      sint_types, all_dims, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
