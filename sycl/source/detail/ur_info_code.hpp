//==--------------------------- ur_info_code.hpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/info_desc_helpers.hpp>

#include <unified-runtime/ur_api.h>

namespace sycl {
inline namespace _V1 {
namespace detail {
// Primary template picks up `T::ur_code` for self-describing traits (those
// carrying `info_class`, `return_type`, `ur_code` members). Explicit
// specializations emitted below from the .def files override the primary
// template for legacy traits. Both forms coexist while migration is in
// progress; SFINAE on `T::ur_code` keeps the primary inert for non-trait
// types so misuse still produces a clean diagnostic.
//
// Self-describing traits without a `ur_code` member (RT-only dispatch) cause
// `UrInfoCode<T>::value` to be undefined; runtime code paths gate on
// `is_ur_dispatched<T>` and never instantiate this primary for those traits.
// `static_assert` here cross-checks ur_code's type against
// `info_class::ur_code_type` to catch wrong-family enum values at compile time.
template <typename T, typename = void> struct UrInfoCode;

template <typename T>
struct UrInfoCode<T, std::void_t<decltype(T::ur_code)>> {
  static_assert(
      !is_self_describing_info_desc<T>::value ||
          std::is_void_v<typename T::info_class::ur_code_type> ||
          std::is_same_v<std::remove_cv_t<decltype(T::ur_code)>,
                         typename T::info_class::ur_code_type>,
      "info-descriptor trait `ur_code` member must match the UR enum type for "
      "its `info_class` family (e.g. ur_device_info_t for info_class::device).");
  static constexpr auto value = T::ur_code;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
