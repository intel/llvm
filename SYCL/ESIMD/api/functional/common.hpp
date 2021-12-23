//===-- common.hpp - This file provides common functions for simd constructors
//      tests -------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Common file for tests on simd class.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/experimental/esimd.hpp>
#include <sycl/sycl.hpp>

#include "../../esimd_test_utils.hpp"
#include "logger.hpp"
#include "type_coverage.hpp"
#include "type_traits.hpp"
#include "value.hpp"

#include <vector>

namespace esimd_test::api::functional {

namespace details {

// Bitwise comparison for two values
template <typename T> bool are_bitwise_equal(T lhs, T rhs) {
  constexpr size_t size{sizeof(T)};

  // Such type-punning is OK from the point of strict aliasing rules
  const auto &lhs_bytes = reinterpret_cast<const unsigned char(&)[size]>(lhs);
  const auto &rhs_bytes = reinterpret_cast<const unsigned char(&)[size]>(rhs);

  bool result{true};
  for (size_t i = 0; i < size; ++i) {
    result &= lhs_bytes[i] == rhs_bytes[i];
  }
  return result;
}

} // namespace details

template <typename DataT>
using shared_allocator = sycl::usm_allocator<DataT, sycl::usm::alloc::shared>;

template <typename DataT>
using shared_vector = std::vector<DataT, shared_allocator<DataT>>;

// A wrapper to speed-up bitwise comparison
template <typename T> bool are_bitwise_equal(T lhs, T rhs) {
  // We are safe to compare unsigned integral types using `==` operator.
  // Still for any other type we might consider the bitwise comparison,
  // including:
  //  - floating-point types, due to nan with opcodes
  //  - signed integer types, to avoid a possibility of UB on trap
  //  representation (negative zero) value access
  if constexpr (std::is_unsigned_v<T>) {
    return lhs == rhs;
  } else {
    return details::are_bitwise_equal(lhs, rhs);
  }
}

} // namespace esimd_test::api::functional
