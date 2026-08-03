//==--------------------------- ur_info_code.hpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/info_desc_traits.hpp>

#include <unified-runtime/ur_api.h>

namespace sycl {
inline namespace _V1 {
namespace detail {
// `UrInfoCode<T>::value` exposes the UR enum used to query an info trait.
// Self-describing traits carry their `ur_code` as a static member; the
// SFINAE specialization below picks it up. Traits without a `ur_code` member
// (RT-only dispatch) leave `UrInfoCode<T>::value` undefined; runtime code
// paths gate on `is_ur_dispatched<T>` and never instantiate this template
// for those traits. `static_assert` cross-checks `ur_code`'s type against
// `info_class::ur_code_type` to catch wrong-family enum values at compile
// time.
template <typename T, typename = void> struct UrInfoCode;

template <typename T> struct UrInfoCode<T, std::void_t<decltype(T::ur_code)>> {
  static_assert(
      !is_self_describing_info_desc<T>::value ||
          std::is_void_v<typename T::info_class::ur_code_type> ||
          std::is_same_v<std::remove_cv_t<decltype(T::ur_code)>,
                         typename T::info_class::ur_code_type>,
      "info-descriptor trait `ur_code` member must match the UR enum type for "
      "its `info_class` family (e.g. ur_device_info_t for "
      "info_class::device).");
  static constexpr auto value = T::ur_code;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
