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

// Class used as a kernel ID.
template <typename DataT, int NumElems, typename...> struct Kernel;

template <typename DataT>
using shared_allocator = sycl::usm_allocator<DataT, sycl::usm::alloc::shared>;

template <typename DataT>
using shared_vector = std::vector<DataT, shared_allocator<DataT>>;

// Provides verification that provided device has necessary aspects to interact
// with current data type.
template <typename T>
inline bool should_skip_test_with(const sycl::device &device) {
  if constexpr (std::is_same_v<T, sycl::half>) {
    if (!device.has(sycl::aspect::fp16)) {
      // TODO: Use TestDescription after removal of the macro
      // ESIMD_TESTS_DISABLE_DEPRECATED_TEST_DESCRIPTION_FOR_LOGS
      log::print_line(
          "Device does not support half precision floating point operations");
      return true;
    }
  } else if constexpr (std::is_same_v<T, double>) {
    if (!device.has(sycl::aspect::fp64)) {
      log::print_line(
          "Device does not support double precision floating point operations");
      return true;
    }
  } else {
    // Suppress compilation warnings
    static_cast<void>(device);
  }

  return false;
}

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
