//===-- functions_1d_select.hpp - Functions for tests on simd rvalue select
//      function. ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functions for tests on simd 1d select function.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "../shared_element.hpp"
#include "common.hpp"

// for std::numeric_limits
#include <limits>
// for std::iota
#include <numeric>

namespace esimd_test::api::functional::functions {

using use_offset = std::true_type;
using do_not_use_offset = std::true_type;

// Descriptor class for the case of calling simd<T,N>::select function.
struct select_rval {
  static std::string get_description() { return "select rvalue"; }

  template <typename DataT, int NumElems, int NumSelectedElems, int Stride>
  static bool call_operator(const DataT *const initial_data,
                            const DataT *const data_for_change,
                            DataT *const out, size_t offset) {
    esimd::simd<DataT, NumElems> simd_obj;
    simd_obj.copy_from(initial_data);
    auto select_result =
        simd_obj.template select<NumSelectedElems, Stride>(offset);

    for (size_t i = 0; i < NumSelectedElems; ++i) {
      select_result[i] = data_for_change[i];
    }
    simd_obj.copy_to(out);

    return std::is_same_v<
        decltype(select_result),
        esimd::simd_view<esimd::simd<DataT, NumElems>,
                         esimd::region1d_t<DataT, NumSelectedElems, Stride>>>;
  }
};

// Descriptor class for the case of calling simd<T,N>::select function.
struct select_lval {
  static std::string get_description() { return "select lvalue"; }

  template <typename DataT, int NumElems, int NumSelectedElems, int Stride>
  static bool call_operator(const DataT *const initial_data,
                            const DataT *const data_for_change,
                            DataT *const out, size_t offset) {
    esimd::simd<DataT, NumElems> src_simd_obj;
    src_simd_obj.copy_from(initial_data);

    esimd::simd<DataT, NumSelectedElems> simd_for_change_values;
    simd_for_change_values.copy_from(data_for_change);

    src_simd_obj.template select<NumSelectedElems, Stride>(offset) =
        simd_for_change_values;
    src_simd_obj.copy_to(out);

    return true;
  }
};

// Descriptor class for the case of calling simd<T,N>::select function.
struct select_simd_view_rval {
  static std::string get_description() { return "select simd view rvalue"; }

  template <typename DataT, int NumElems, int NumSelectedElems, int Stride>
  static bool call_operator(const DataT *const initial_data,
                            const DataT *const data_for_change,
                            DataT *const out, size_t offset) {
    esimd::simd<DataT, NumElems> src_simd_obj;
    src_simd_obj.copy_from(initial_data);

    auto simd_view_instance = src_simd_obj.template bit_cast_view<DataT>();

    auto selected_elems =
        simd_view_instance.template select<NumSelectedElems, Stride>(offset);

    for (size_t i = 0; i < NumSelectedElems; ++i) {
      selected_elems[i] = data_for_change[i];
    }

    src_simd_obj.copy_to(out);

    return true;
  }
};

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename TestCaseT, typename NumSelectedElemsT, typename StrideT,
          typename OffsetT, typename DataT, typename DimT>
class run_test {
  static constexpr int NumElems = DimT::value;
  static constexpr int NumSelectedElems = NumSelectedElemsT::value;
  static constexpr int Stride = StrideT::value;
  static constexpr int Offset = OffsetT::value;
  using TestDescriptionT = TestDescription<NumElems, TestCaseT>;

public:
  bool operator()(sycl::queue &queue, const std::string &data_type) {
    log::trace<TestDescriptionT>(data_type);
    static_assert(NumElems >= NumSelectedElems * Stride + Offset &&
                  "Number selected elements should be less than simd size.");
    bool passed = true;
    size_t alignment_value = alignof(DataT);

    constexpr size_t value_for_increase_ref_data_for_change = 50;
    static_assert(std::numeric_limits<signed char>::max() >
                  value_for_increase_ref_data_for_change + NumElems);

    shared_allocator<DataT> allocator(queue);
    shared_vector<DataT> result(NumElems, allocator);
    shared_vector<DataT> initial_ref_data(NumElems, allocator);
    shared_vector<DataT> ref_data_for_change(NumElems, allocator);

    shared_element<bool> is_correct_type(queue, true);

    std::iota(initial_ref_data.begin(), initial_ref_data.end(), 0);
    // We should have different values in the first reference data and in the
    // second reference data.
    std::iota(ref_data_for_change.begin(), ref_data_for_change.end(),
              initial_ref_data.back() + value_for_increase_ref_data_for_change);

    queue.submit([&](sycl::handler &cgh) {
      DataT *init_ref_ptr = initial_ref_data.data();
      DataT *ref_data_for_change_ptr = ref_data_for_change.data();
      DataT *const out_ptr = result.data();
      auto is_correct_type_ptr = is_correct_type.data();

      cgh.single_task<
          Kernel<DataT, NumElems, TestCaseT, NumSelectedElemsT, StrideT>>(
          [=]() SYCL_ESIMD_KERNEL {
            *is_correct_type_ptr =
                TestCaseT::template call_operator<DataT, NumElems,
                                                  NumSelectedElems, Stride>(
                    init_ref_ptr, ref_data_for_change_ptr, out_ptr, Offset);
          });
    });
    queue.wait_and_throw();

    std::vector<size_t> selected_indexes;
    // Collect the indexess that has been selected.
    for (size_t i = Offset; i < Stride * NumSelectedElems + Offset;
         i += Stride) {
      selected_indexes.push_back(i);
    }

    // Push the largest value to avoid the following error: can't dereference
    // out of range vector iterator.
    selected_indexes.push_back(std::numeric_limits<size_t>::max());
    auto next_selected_index = selected_indexes.begin();

    // Verify that values, that do not was selected has initial values.
    for (size_t i = 0; i < NumElems; ++i) {
      // If current index is less than selected index verify that this element
      // hasn't been selected and changed.
      if (i < *next_selected_index) {
        const DataT &expected = initial_ref_data[i];
        const DataT &retrieved = result[i];
        if (expected != retrieved) {
          passed = fail_test(i, expected, retrieved, data_type);
        }
      } else {
        const DataT &expected = ref_data_for_change[(i - Offset) / Stride];
        const DataT &retrieved = result[i];
        if (expected != retrieved) {
          passed = fail_test(i, expected, retrieved, data_type);
        }
        next_selected_index++;
      }
    }

    if (!is_correct_type.value()) {
      passed = false;
      log::fail(TestDescriptionT(data_type), "Unexpected return type.");
    }

    return passed;
  }

private:
  bool fail_test(size_t i, DataT expected, DataT retrieved,
                 const std::string &data_type) {
    log::fail(TestDescriptionT(data_type), "Unexpected value at index ", i,
              ", retrieved: ", retrieved, ", expected: ", expected,
              ", with size selected elems: ", NumSelectedElems,
              ", with stride: ", Stride, ", with offset: ", Offset);

    return false;
  }
};

// Aliases to provide size or stride values to test.
// This is the syntax sugar just for code readability.
template <int N> using stride_type = std::integral_constant<int, N>;
template <int N> using size_type = std::integral_constant<int, N>;
template <int N> using offset_type = std::integral_constant<int, N>;

template <typename SelectT, int NumSelectedElems, int Stride, int Offset,
          typename... ArgsT>
bool run_with_size_stride_offset(ArgsT &&...args) {
  bool passed =
      for_all_combinations<run_test, SelectT, size_type<NumSelectedElems>,
                           stride_type<Stride>, offset_type<Offset>>(
          std::forward<ArgsT>(args)...);

  return passed;
}

template <tested_types TestedTypes, typename SelectT>
bool run_test_for_types(sycl::queue &queue) {
  bool passed = true;
  constexpr int desired_simd_small_size = 1;
  constexpr int desired_simd_large_size = 16;
  constexpr int coefficient_of_division = 3;
  constexpr int zero_offset_value = 0;
  constexpr int small_offset_value = 1;
  constexpr int large_offset_value =
      desired_simd_large_size -
      round_up_int_division(2 * desired_simd_large_size, 3);

  const auto small_size = get_dimensions<desired_simd_small_size>();
  const auto great_size = get_dimensions<desired_simd_large_size>();

#if SIMD_RUN_TEST_WITH_CHAR_TYPES
  const auto types = get_tested_types<TestedTypes>();
#else
  const auto types =
      named_type_pack<short, unsigned short, int, unsigned int, long,
                      unsigned long, float, long long,
                      unsigned long long>::generate("short", "unsigned short",
                                                    "int", "unsigned int",
                                                    "long", "unsigned long",
                                                    "float", "long long",
                                                    "unsigned long long");
#endif

  // Checks are run for specific combinations of types, sizes, strides and
  // offsets.
  passed &= run_with_size_stride_offset<SelectT, 1, 1, zero_offset_value>(
      types, small_size, queue);

  passed &= run_with_size_stride_offset<
      SelectT, desired_simd_large_size / coefficient_of_division,
      coefficient_of_division, zero_offset_value>(types, great_size, queue);

  passed &= run_with_size_stride_offset<
      SelectT, desired_simd_large_size / coefficient_of_division,
      coefficient_of_division, zero_offset_value>(types, great_size, queue);

  passed &= run_with_size_stride_offset<
      SelectT, coefficient_of_division,
      desired_simd_large_size / coefficient_of_division, zero_offset_value>(
      types, great_size, queue);

  passed &=
      run_with_size_stride_offset<SelectT,
                                  desired_simd_large_size - small_offset_value,
                                  desired_simd_small_size, small_offset_value>(
          types, great_size, queue);

  passed &= run_with_size_stride_offset<
      SelectT, desired_simd_large_size / coefficient_of_division, 2,
      large_offset_value>(types, great_size, queue);

  return passed;
}

} // namespace esimd_test::api::functional::functions
