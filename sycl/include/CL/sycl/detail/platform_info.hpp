//==------ platform_info.hpp - SYCL platform info methods ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/common_info.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/info/info_desc.hpp>

namespace cl {
namespace sycl {
namespace detail {

// The platform information methods
template <typename T, info::platform param> struct get_platform_info {};

template <info::platform param>
struct get_platform_info<string_class, param> {
  static string_class _(RT::PiPlatform plt) {
    size_t resultSize;
    // TODO catch an exception and put it to list of asynchronous exceptions
    PI_CALL(RT::piPlatformGetInfo,
      plt, pi::cast<pi_platform_info>(param), 0, nullptr, &resultSize);
    if (resultSize == 0) {
      return "";
    }
    unique_ptr_class<char[]> result(new char[resultSize]);
    // TODO catch an exception and put it to list of asynchronous exceptions
    PI_CALL(RT::piPlatformGetInfo,
      plt, pi::cast<pi_platform_info>(param), resultSize, result.get(), nullptr);
    return result.get();
  }
};

template <>
struct get_platform_info<vector_class<string_class>,
                         info::platform::extensions> {
  static vector_class<string_class> _(RT::PiPlatform plt) {
    string_class result =
        get_platform_info<string_class, info::platform::extensions>::_(plt);
    return split_string(result, ' ');
  }
};

// Host platform information methods
template <info::platform param>
typename info::param_traits<info::platform, param>::return_type
get_platform_info_host() = delete;

template <> string_class get_platform_info_host<info::platform::profile>();

template <> string_class get_platform_info_host<info::platform::version>();

template <> string_class get_platform_info_host<info::platform::name>();

template <> string_class get_platform_info_host<info::platform::vendor>();

template <>
vector_class<string_class> get_platform_info_host<info::platform::extensions>();

} // namespace detail
} // namespace sycl
} // namespace cl
