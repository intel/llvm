//==------------------------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/kernel_bundle.hpp>

namespace sycl {
inline namespace _V1 {

namespace detail {
struct link_props;
} // namespace detail

namespace ext::oneapi::experimental {

/////////////////////////
// PropertyT syclex::fast_link
/////////////////////////
struct fast_link
    : detail::run_time_property_key<fast_link, detail::PropKind::FastLink> {
  fast_link(bool DoFastLink = true) : value(DoFastLink) {}

  bool value;
};
using fast_link_key = fast_link;

template <>
struct is_property_key_of<fast_link_key, sycl::detail::link_props>
    : std::true_type {};
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
