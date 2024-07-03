//==-------- kernel_info.hpp - SYCL kernel info methods --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/error_handling/error_handling.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/info_desc_helpers.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/device.hpp>
#include <sycl/info/info_desc.hpp>

#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace detail {

template <typename Param>
typename std::enable_if<
    std::is_same<typename Param::return_type, std::string>::value,
    std::string>::type
get_kernel_info(sycl::detail::pi::PiKernel Kernel, const PluginPtr &Plugin) {
  static_assert(detail::is_kernel_info_desc<Param>::value,
                "Invalid kernel information descriptor");
  size_t ResultSize = 0;

  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin->call<PiApiKind::piKernelGetInfo>(Kernel, PiInfoCode<Param>::value, 0,
                                           nullptr, &ResultSize);
  if (ResultSize == 0) {
    return "";
  }
  std::vector<char> Result(ResultSize);
  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin->call<PiApiKind::piKernelGetInfo>(Kernel, PiInfoCode<Param>::value,
                                           ResultSize, Result.data(), nullptr);
  return std::string(Result.data());
}

template <typename Param>
typename std::enable_if<
    std::is_same<typename Param::return_type, uint32_t>::value, uint32_t>::type
get_kernel_info(sycl::detail::pi::PiKernel Kernel, const PluginPtr &Plugin) {
  uint32_t Result = 0;

  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin->call<PiApiKind::piKernelGetInfo>(Kernel, PiInfoCode<Param>::value,
                                           sizeof(uint32_t), &Result, nullptr);
  return Result;
}

// Device-specific methods
template <typename Param>
typename std::enable_if<IsSubGroupInfo<Param>::value>::type
get_kernel_device_specific_info_helper(sycl::detail::pi::PiKernel Kernel,
                                       sycl::detail::pi::PiDevice Device,
                                       const PluginPtr &Plugin, void *Result,
                                       size_t Size) {
  Plugin->call<PiApiKind::piKernelGetSubGroupInfo>(
      Kernel, Device, PiInfoCode<Param>::value, 0, nullptr, Size, Result,
      nullptr);
}

template <typename Param>
typename std::enable_if<!IsSubGroupInfo<Param>::value>::type
get_kernel_device_specific_info_helper(sycl::detail::pi::PiKernel Kernel,
                                       sycl::detail::pi::PiDevice Device,
                                       const PluginPtr &Plugin, void *Result,
                                       size_t Size) {
  sycl::detail::pi::PiResult Error =
      Plugin->call_nocheck<PiApiKind::piKernelGetGroupInfo>(
          Kernel, Device, PiInfoCode<Param>::value, Size, Result, nullptr);
  if (Error != PI_SUCCESS)
    kernel_get_group_info::handleErrorOrWarning(Error, PiInfoCode<Param>::value,
                                                Plugin);
}

template <typename Param>
typename std::enable_if<
    !std::is_same<typename Param::return_type, sycl::range<3>>::value,
    typename Param::return_type>::type
get_kernel_device_specific_info(sycl::detail::pi::PiKernel Kernel,
                                sycl::detail::pi::PiDevice Device,
                                const PluginPtr &Plugin) {
  static_assert(is_kernel_device_specific_info_desc<Param>::value,
                "Unexpected kernel_device_specific information descriptor");
  typename Param::return_type Result = {};
  // TODO catch an exception and put it to list of asynchronous exceptions
  get_kernel_device_specific_info_helper<Param>(
      Kernel, Device, Plugin, &Result, sizeof(typename Param::return_type));
  return Result;
}

template <typename Param>
typename std::enable_if<
    std::is_same<typename Param::return_type, sycl::range<3>>::value,
    sycl::range<3>>::type
get_kernel_device_specific_info(sycl::detail::pi::PiKernel Kernel,
                                sycl::detail::pi::PiDevice Device,
                                const PluginPtr &Plugin) {
  static_assert(is_kernel_device_specific_info_desc<Param>::value,
                "Unexpected kernel_device_specific information descriptor");
  size_t Result[3] = {0, 0, 0};
  // TODO catch an exception and put it to list of asynchronous exceptions
  get_kernel_device_specific_info_helper<Param>(Kernel, Device, Plugin, Result,
                                                sizeof(size_t) * 3);
  return sycl::range<3>(Result[0], Result[1], Result[2]);
}

// TODO: This is used by a deprecated version of
// info::kernel_device_specific::max_sub_group_size taking an input paramter.
// This should be removed when the deprecated info query is removed.
template <typename Param>
uint32_t get_kernel_device_specific_info_with_input(
    sycl::detail::pi::PiKernel Kernel, sycl::detail::pi::PiDevice Device,
    sycl::range<3> In, const PluginPtr &Plugin) {
  static_assert(is_kernel_device_specific_info_desc<Param>::value,
                "Unexpected kernel_device_specific information descriptor");
  static_assert(std::is_same<typename Param::return_type, uint32_t>::value,
                "Unexpected return type");
  static_assert(IsSubGroupInfo<Param>::value,
                "Unexpected kernel_device_specific information descriptor for "
                "query with input");
  size_t Input[3] = {In[0], In[1], In[2]};
  uint32_t Result = 0;
  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin->call<PiApiKind::piKernelGetSubGroupInfo>(
      Kernel, Device, PiInfoCode<Param>::value, sizeof(size_t) * 3, Input,
      sizeof(uint32_t), &Result, nullptr);

  return Result;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
