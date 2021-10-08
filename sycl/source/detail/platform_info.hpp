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
#include <detail/plugin.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

// The platform information methods
template <typename T, info::platform param> struct get_platform_info {};

template <info::platform param> struct get_platform_info<std::string, param> {
  static std::string get(RT::PiPlatform plt, const plugin &Plugin) {
    size_t resultSize;
    // TODO catch an exception and put it to list of asynchronous exceptions
    Plugin.call<PiApiKind::piPlatformGetInfo>(
        plt, pi::cast<pi_platform_info>(param), 0, nullptr, &resultSize);
    if (resultSize == 0) {
      return "";
    }
    std::unique_ptr<char[]> result(new char[resultSize]);
    // TODO catch an exception and put it to list of asynchronous exceptions
    Plugin.call<PiApiKind::piPlatformGetInfo>(
        plt, pi::cast<pi_platform_info>(param), resultSize, result.get(),
        nullptr);
    return result.get();
  }
};

template <>
struct get_platform_info<std::vector<std::string>, info::platform::extensions> {
  static std::vector<std::string> get(RT::PiPlatform plt,
                                      const plugin &Plugin) {
    std::string result =
        get_platform_info<std::string, info::platform::extensions>::get(plt,
                                                                        Plugin);
    return split_string(result, ' ');
  }
};

// Host platform information methods
template <info::platform param>
inline typename info::param_traits<info::platform, param>::return_type
get_platform_info_host() = delete;

template <>
inline std::string get_platform_info_host<info::platform::profile>() {
  return "FULL PROFILE";
}

template <>
inline std::string get_platform_info_host<info::platform::version>() {
  return "1.2";
}

template <> inline std::string get_platform_info_host<info::platform::name>() {
  return "SYCL host platform";
}

template <>
inline std::string get_platform_info_host<info::platform::vendor>() {
  return "";
}

template <>
inline std::vector<std::string>
get_platform_info_host<info::platform::extensions>() {
  // TODO update when appropriate
  return {};
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
