//==----- platform.hpp - SYCL platform information descriptors -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp> // for __SYCL2020_DEPRECATED
#include <sycl/detail/info_desc_traits.hpp>
#include <unified-runtime/ur_api.h>

#include <string>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace info {
// A.1 Platform information desctiptors
namespace platform {
template <ur_platform_info_t UrCode>
using platform_traits =
    sycl::detail::ur_traits_base<sycl::detail::info_class::platform, UrCode>;

// TODO Despite giving this deprecation warning, we're still yet to implement
// info::device::aspects.
struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020, use device::get_info() "
                             "with info::device::aspects instead") extensions
    : platform_traits<UR_PLATFORM_INFO_EXTENSIONS> {
  using return_type = std::vector<std::string>;
};
struct profile : platform_traits<UR_PLATFORM_INFO_PROFILE> {
  using return_type = std::string;
};
struct version : platform_traits<UR_PLATFORM_INFO_VERSION> {
  using return_type = std::string;
};
struct name : platform_traits<UR_PLATFORM_INFO_NAME> {
  using return_type = std::string;
};
struct vendor : platform_traits<UR_PLATFORM_INFO_VENDOR_NAME> {
  using return_type = std::string;
};
} // namespace platform
} // namespace info

namespace detail {
// SFINAE predicate confining `platform::get_info<T>()` to platform traits.
// `return_type` alias is load-bearing for ABI symbol mangling — keep stable.
template <typename T>
struct is_platform_info_desc : is_info_desc_for<T, info_class::platform> {};
} // namespace detail
} // namespace _V1
} // namespace sycl
