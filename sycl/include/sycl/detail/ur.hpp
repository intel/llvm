//==---------- ur.hpp - Unified Runtime integration helpers ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file
///
/// C++ utilities for Unified Runtime integration.
///
/// \ingroup sycl_ur

#pragma once

#include <sycl/detail/export.hpp>
#include <ur_api.h>

#include <type_traits>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {
namespace ur {
// Return true if we want to trace UR related activities.
bool trace();

// Report error and no return (keeps compiler happy about no return statements).
[[noreturn]] __SYCL_EXPORT void die(const char *Message);

__SYCL_EXPORT void assertion(bool Condition, const char *Message = nullptr);

// Want all the needed casts be explicit, do not define conversion operators.
template <class To, class From> To cast(From value);

// Want all the needed casts be explicit, do not define conversion
// operators.
template <class To, class From> inline To cast(From value) {
  // TODO: see if more sanity checks are possible.
  assertion(sizeof(From) == sizeof(To), "assert: cast failed size check");
  return (To)(value);
}

// Helper traits for identifying std::vector with arbitrary element type.
template <typename T> struct IsStdVector : std::false_type {};
template <typename T> struct IsStdVector<std::vector<T>> : std::true_type {};

// Overload for vectors that applies the cast to all elements. This
// creates a new vector.
template <class To, class FromE> To cast(std::vector<FromE> Values) {
  static_assert(IsStdVector<To>::value, "Return type must be a vector.");
  To ResultVec;
  ResultVec.reserve(Values.size());
  for (FromE &Val : Values) {
    ResultVec.push_back(cast<typename To::value_type>(Val));
  }
  return ResultVec;
}

} // namespace ur
} // namespace detail
} // namespace _V1
} // namespace sycl
