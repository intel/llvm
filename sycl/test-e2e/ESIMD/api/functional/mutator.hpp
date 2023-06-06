//===-- mutator.hpp - This file provides common function and classes to mutate
//      reference data. ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides common function and classes to mutate reference data.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "type_traits.hpp"
#include "value.hpp"
#include <sycl/sycl.hpp>

// for std::for_each
#include <algorithm>
#include <vector>

namespace esimd_test::api::functional {

// In some test cases it’s possible to pass a reference data as the input data
// directly without any modification. For example, if we check some copy
// constructor or memory move operation it’s OK to have any data values as the
// input ones. But in most cases we need to consider the possibility of UB for
// C++ operations due to modification of input values.
//
// Mutation mechanism covers such requirement.
// The mutator namespace is intended to store such mutators alongside with
// the generic ones provided below.
namespace mutator {

// Replace specific reference values to the bigger ones, so we can safely
// substract `m_value` later.
template <typename T> class For_subtraction {
  T m_value;

public:
  For_subtraction(T val)
      : m_value((assert(val > 0 && "Invalid value."), val)) {}

  void operator()(T &val) {
    if constexpr (type_traits::is_sycl_signed_v<T>) {
      const T lower_border = value<T>::lowest() + m_value;
      if (val < lower_border) {
        // Validate only the negative infinity case, as the NaN will not enter
        // this branch.
        if constexpr (type_traits::is_sycl_floating_point_v<T>) {
          // sycl::half will be converted to float
          if (std::isinf(val)) {
            return;
          }
        }
        // we need to update value to avoid UB during subtraction.
        val = lower_border;
      }
    }
  }
};

// Replace specific reference values to the smaller ones, so we can safely add
// `m_value` later.
template <typename T> class For_addition {
  T m_value;

public:
  For_addition(T val) : m_value((assert(val > 0 && "Invalid value."), val)) {}

  void operator()(T &val) {
    if constexpr (type_traits::is_sycl_signed_v<T>) {
      const T upper_border = value<T>::max() - m_value;
      if (val > upper_border) {
        // Validate only the negative infinity case, as the NaN will not enter
        // this branch.
        if constexpr (type_traits::is_sycl_floating_point_v<T>) {
          // sycl::half will be converted to float
          if (std::isinf(val)) {
            return;
          }
        }
        // we need to update value to avoid UB during addition.
        val = upper_border;
      }
    }
  }
};

// Replace specific reference values to the divided ones, so we can safely
// multiply to `m_value` later.
template <typename T> class For_multiplication {
  T m_value;

public:
  For_multiplication(T val)
      : m_value((assert(val > 0 && "Invalid value."), val)) {}

  void operator()(T &val) {
    // We don't need to change the inf values, because inf * value gives inf.
    if constexpr (type_traits::is_sycl_floating_point_v<T>) {
      if (val == value<T>::inf() || val == -value<T>::inf()) {
        return;
      }
    }

    const T upper_border = value<T>::max() / m_value;
    // we need to update value to avoid UB during multiplication for positive
    // and negative numbers.
    if (val > upper_border) {
      val = upper_border;
    } else if (val < -upper_border) {
      val = -upper_border;
    }
  }
};

} // namespace mutator

// Applies provided mutator to each value for provided container.
template <typename T, typename MutatorT>
inline void mutate(std::vector<T> &input_vector, MutatorT &&mutator) {
  std::for_each(input_vector.begin(), input_vector.end(), mutator);
}

} // namespace esimd_test::api::functional
