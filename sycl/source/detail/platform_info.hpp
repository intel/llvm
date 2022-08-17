//==------ platform_info.hpp - SYCL platform info methods ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <detail/plugin.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/common_info.hpp>
#include <sycl/detail/info_desc_helpers.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/info/info_desc.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

inline std::string get_platform_info_string_impl(RT::PiPlatform Plt,
                                                 const plugin &Plugin,
                                                 pi_platform_info PiCode) {
  size_t ResultSize;
  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin.call<PiApiKind::piPlatformGetInfo>(Plt, PiCode, 0, nullptr,
                                            &ResultSize);
  if (ResultSize == 0) {
    return "";
  }
  std::unique_ptr<char[]> Result(new char[ResultSize]);
  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin.call<PiApiKind::piPlatformGetInfo>(Plt, PiCode, ResultSize,
                                            Result.get(), nullptr);
  return Result.get();
}
// The platform information methods
template <typename Param>
typename std::enable_if<
    std::is_same<typename Param::return_type, std::string>::value,
    std::string>::type
get_platform_info(RT::PiPlatform Plt, const plugin &Plugin) {
  static_assert(is_platform_info_desc<Param>::value,
                "Invalid platform information descriptor");
  return get_platform_info_string_impl(Plt, Plugin,
                                       detail::PiInfoCode<Param>::value);
}

template <typename Param>
typename std::enable_if<std::is_same<Param, info::platform::extensions>::value,
                        std::vector<std::string>>::type
get_platform_info(RT::PiPlatform Plt, const plugin &Plugin) {
  static_assert(is_platform_info_desc<Param>::value,
                "Invalid platform information descriptor");
  std::string Result = get_platform_info_string_impl(
      Plt, Plugin, detail::PiInfoCode<info::platform::extensions>::value);
  return split_string(Result, ' ');
}

// Host platform information methods
template <typename Param>
inline typename Param::return_type get_platform_info_host() = delete;

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
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
