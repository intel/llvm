//==--- use_root_sync_prop.hpp --- SYCL extension for root groups ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Declaration of the property to be included in <sycl/handler.hpp> without
// pulling in the entire root group extension header there.

#pragma once

#include <sycl/ext/oneapi/properties/properties.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

struct use_root_sync_key
    : detail::compile_time_property_key<detail::PropKind::UseRootSync> {
  using value_t = property_value<use_root_sync_key>;
};

inline constexpr use_root_sync_key::value_t use_root_sync;

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
