//==-- cluster_group_prop.hpp --- SYCL extension for event mode property ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

enum class event_mode_enum { none, low_power };

struct event_mode
    : detail::run_time_property_key<event_mode, detail::PropKind::EventMode> {
  event_mode(event_mode_enum mode) : value(mode) {}

  event_mode_enum value;
};

using event_mode_key = event_mode;

inline bool operator==(const event_mode &lhs, const event_mode &rhs) {
  return lhs.value == rhs.value;
}
inline bool operator!=(const event_mode &lhs, const event_mode &rhs) {
  return !(lhs == rhs);
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
