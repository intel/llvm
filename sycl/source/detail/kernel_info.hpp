//==-------- kernel_info.hpp - SYCL kernel info methods --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/common.hpp>
#include <sycl/detail/common_info.hpp>
#include <sycl/detail/info_desc_helpers.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/device.hpp>
#include <sycl/info/info_desc.hpp>

#include <type_traits>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

template <typename Param>
typename std::enable_if<
    std::is_same<typename Param::return_type, std::string>::value,
    std::string>::type
get_kernel_info(RT::PiKernel Kernel, const plugin &Plugin) {
  static_assert(detail::is_kernel_info_desc<Param>::value,
                "Invalid kernel information descriptor");
  size_t ResultSize;

  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin.call<PiApiKind::piKernelGetInfo>(Kernel, PiInfoCode<Param>::value, 0,
                                          nullptr, &ResultSize);
  if (ResultSize == 0) {
    return "";
  }
  std::vector<char> Result(ResultSize);
  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin.call<PiApiKind::piKernelGetInfo>(Kernel, PiInfoCode<Param>::value,
                                          ResultSize, Result.data(), nullptr);
  return std::string(Result.data());
}

template <typename Param>
typename std::enable_if<
    std::is_same<typename Param::return_type, uint32_t>::value, uint32_t>::type
get_kernel_info(RT::PiKernel Kernel, const plugin &Plugin) {
  uint32_t Result;

  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin.call<PiApiKind::piKernelGetInfo>(Kernel, PiInfoCode<Param>::value,
                                          sizeof(uint32_t), &Result, nullptr);
  return Result;
}

// Device-specific methods
template <typename Param>
typename std::enable_if<IsSubGroupInfo<Param>::value>::type
get_kernel_device_specific_info_helper(RT::PiKernel Kernel, RT::PiDevice Device,
                                       const plugin &Plugin, void *Result,
                                       size_t Size) {
  Plugin.call<PiApiKind::piKernelGetSubGroupInfo>(
      Kernel, Device, PiInfoCode<Param>::value, 0, nullptr, Size, Result,
      nullptr);
}

template <typename Param>
typename std::enable_if<!IsSubGroupInfo<Param>::value>::type
get_kernel_device_specific_info_helper(RT::PiKernel Kernel, RT::PiDevice Device,
                                       const plugin &Plugin, void *Result,
                                       size_t Size) {
  Plugin.call<PiApiKind::piKernelGetGroupInfo>(
      Kernel, Device, PiInfoCode<Param>::value, Size, Result, nullptr);
}

template <typename Param>
typename std::enable_if<
    !std::is_same<typename Param::return_type, sycl::range<3>>::value,
    typename Param::return_type>::type
get_kernel_device_specific_info(RT::PiKernel Kernel, RT::PiDevice Device,
                                const plugin &Plugin) {
  static_assert(is_kernel_device_specific_info_desc<Param>::value,
                "Unexpected kernel_device_specific information descriptor");
  typename Param::return_type Result;
  // TODO catch an exception and put it to list of asynchronous exceptions
  get_kernel_device_specific_info_helper<Param>(
      Kernel, Device, Plugin, &Result, sizeof(typename Param::return_type));
  return Result;
}

template <typename Param>
typename std::enable_if<
    std::is_same<typename Param::return_type, sycl::range<3>>::value,
    sycl::range<3>>::type
get_kernel_device_specific_info(RT::PiKernel Kernel, RT::PiDevice Device,
                                const plugin &Plugin) {
  static_assert(is_kernel_device_specific_info_desc<Param>::value,
                "Unexpected kernel_device_specific information descriptor");
  size_t Result[3];
  // TODO catch an exception and put it to list of asynchronous exceptions
  get_kernel_device_specific_info_helper<Param>(Kernel, Device, Plugin, Result,
                                                sizeof(size_t) * 3);
  return sycl::range<3>(Result[0], Result[1], Result[2]);
}

// TODO: This is used by a deprecated version of
// info::kernel_device_specific::max_sub_group_size taking an input paramter.
// This should be removed when the deprecated info query is removed.
template <typename Param>
uint32_t get_kernel_device_specific_info_with_input(RT::PiKernel Kernel,
                                                    RT::PiDevice Device,
                                                    sycl::range<3> In,
                                                    const plugin &Plugin) {
  static_assert(is_kernel_device_specific_info_desc<Param>::value,
                "Unexpected kernel_device_specific information descriptor");
  static_assert(std::is_same<typename Param::return_type, uint32_t>::value,
                "Unexpected return type");
  static_assert(IsSubGroupInfo<Param>::value,
                "Unexpected kernel_device_specific information descriptor for "
                "query with input");
  size_t Input[3] = {In[0], In[1], In[2]};
  uint32_t Result;
  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin.call<PiApiKind::piKernelGetSubGroupInfo>(
      Kernel, Device, PiInfoCode<Param>::value, sizeof(size_t) * 3, Input,
      sizeof(uint32_t), &Result, nullptr);

  return Result;
}

template <typename Param>
inline typename Param::return_type
get_kernel_device_specific_info_host(const sycl::device &Device) = delete;

template <>
inline sycl::range<3> get_kernel_device_specific_info_host<
    info::kernel_device_specific::global_work_size>(const sycl::device &) {
  throw invalid_object_error("This instance of kernel is a host instance",
                             PI_ERROR_INVALID_KERNEL);
}

template <>
inline size_t get_kernel_device_specific_info_host<
    info::kernel_device_specific::work_group_size>(const sycl::device &Dev) {
  return Dev.get_info<info::device::max_work_group_size>();
}

template <>
inline sycl::range<3> get_kernel_device_specific_info_host<
    info::kernel_device_specific::compile_work_group_size>(
    const sycl::device &) {
  return {0, 0, 0};
}

template <>
inline size_t get_kernel_device_specific_info_host<
    info::kernel_device_specific::preferred_work_group_size_multiple>(
    const sycl::device &Dev) {
  return get_kernel_device_specific_info_host<
      info::kernel_device_specific::work_group_size>(Dev);
}

template <>
inline size_t get_kernel_device_specific_info_host<
    info::kernel_device_specific::private_mem_size>(const sycl::device &) {
  return 0;
}

template <>
inline uint32_t get_kernel_device_specific_info_host<
    info::kernel_device_specific::ext_codeplay_num_regs>(const sycl::device &) {
  return 0;
}

template <>
inline uint32_t get_kernel_device_specific_info_host<
    info::kernel_device_specific::max_num_sub_groups>(const sycl::device &) {
  throw invalid_object_error("This instance of kernel is a host instance",
                             PI_ERROR_INVALID_KERNEL);
}

template <>
inline uint32_t get_kernel_device_specific_info_host<
    info::kernel_device_specific::max_sub_group_size>(const sycl::device &) {
  throw invalid_object_error("This instance of kernel is a host instance",
                             PI_ERROR_INVALID_KERNEL);
}

template <>
inline uint32_t get_kernel_device_specific_info_host<
    info::kernel_device_specific::compile_num_sub_groups>(
    const sycl::device &) {
  throw invalid_object_error("This instance of kernel is a host instance",
                             PI_ERROR_INVALID_KERNEL);
}

template <>
inline uint32_t get_kernel_device_specific_info_host<
    info::kernel_device_specific::compile_sub_group_size>(
    const sycl::device &) {
  throw invalid_object_error("This instance of kernel is a host instance",
                             PI_ERROR_INVALID_KERNEL);
}
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
