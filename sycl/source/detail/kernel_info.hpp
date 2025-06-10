//==-------- kernel_info.hpp - SYCL kernel info methods --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/error_handling/error_handling.hpp>
#include <detail/ur_info_code.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/info_desc_helpers.hpp>
#include <sycl/detail/ur.hpp>
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
get_kernel_info(ur_kernel_handle_t Kernel, const AdapterPtr &Adapter) {
  static_assert(detail::is_kernel_info_desc<Param>::value,
                "Invalid kernel information descriptor");
  size_t ResultSize = 0;

  // TODO catch an exception and put it to list of asynchronous exceptions
  Adapter->call<UrApiKind::urKernelGetInfo>(Kernel, UrInfoCode<Param>::value, 0,
                                            nullptr, &ResultSize);
  if (ResultSize == 0) {
    return "";
  }
  std::vector<char> Result(ResultSize);
  // TODO catch an exception and put it to list of asynchronous exceptions
  Adapter->call<UrApiKind::urKernelGetInfo>(Kernel, UrInfoCode<Param>::value,
                                            ResultSize, Result.data(), nullptr);
  return std::string(Result.data());
}

template <typename Param>
typename std::enable_if<
    std::is_same<typename Param::return_type, uint32_t>::value, uint32_t>::type
get_kernel_info(ur_kernel_handle_t Kernel, const AdapterPtr &Adapter) {
  ur_result_t Result = UR_RESULT_SUCCESS;

  // TODO catch an exception and put it to list of asynchronous exceptions
  Adapter->call<UrApiKind::urKernelGetInfo>(Kernel, UrInfoCode<Param>::value,
                                            sizeof(uint32_t), &Result, nullptr);
  return Result;
}

// Device-specific methods
template <typename Param>
typename std::enable_if<IsSubGroupInfo<Param>::value>::type
get_kernel_device_specific_info_helper(ur_kernel_handle_t Kernel,
                                       ur_device_handle_t Device,
                                       const AdapterPtr &Adapter, void *Result,
                                       size_t Size) {
  Adapter->call<UrApiKind::urKernelGetSubGroupInfo>(
      Kernel, Device, UrInfoCode<Param>::value, Size, Result, nullptr);
}

template <typename Param>
typename std::enable_if<IsKernelInfo<Param>::value>::type
get_kernel_device_specific_info_helper(
    ur_kernel_handle_t Kernel, [[maybe_unused]] ur_device_handle_t Device,
    const AdapterPtr &Adapter, void *Result, size_t Size) {
  Adapter->call<UrApiKind::urKernelGetInfo>(Kernel, UrInfoCode<Param>::value,
                                            Size, Result, nullptr);
}

template <typename Param>
typename std::enable_if<!IsSubGroupInfo<Param>::value &&
                        !IsKernelInfo<Param>::value>::type
get_kernel_device_specific_info_helper(ur_kernel_handle_t Kernel,
                                       ur_device_handle_t Device,
                                       const AdapterPtr &Adapter, void *Result,
                                       size_t Size) {
  ur_result_t Error = Adapter->call_nocheck<UrApiKind::urKernelGetGroupInfo>(
      Kernel, Device, UrInfoCode<Param>::value, Size, Result, nullptr);
  if (Error != UR_RESULT_SUCCESS)
    kernel_get_group_info::handleErrorOrWarning(Error, UrInfoCode<Param>::value,
                                                Adapter);
}

template <typename Param>
typename std::enable_if<
    !std::is_same<typename Param::return_type, sycl::range<3>>::value,
    typename Param::return_type>::type
get_kernel_device_specific_info(ur_kernel_handle_t Kernel,
                                ur_device_handle_t Device,
                                const AdapterPtr &Adapter) {
  static_assert(is_kernel_device_specific_info_desc<Param>::value,
                "Unexpected kernel_device_specific information descriptor");
  typename Param::return_type Result = {};
  // TODO catch an exception and put it to list of asynchronous exceptions
  get_kernel_device_specific_info_helper<Param>(
      Kernel, Device, Adapter, &Result, sizeof(typename Param::return_type));
  return Result;
}

template <typename Param>
typename std::enable_if<
    std::is_same<typename Param::return_type, sycl::range<3>>::value,
    sycl::range<3>>::type
get_kernel_device_specific_info(ur_kernel_handle_t Kernel,
                                ur_device_handle_t Device,
                                const AdapterPtr &Adapter) {
  static_assert(is_kernel_device_specific_info_desc<Param>::value,
                "Unexpected kernel_device_specific information descriptor");
  size_t Result[3] = {0, 0, 0};
  // TODO catch an exception and put it to list of asynchronous exceptions
  get_kernel_device_specific_info_helper<Param>(Kernel, Device, Adapter, Result,
                                                sizeof(size_t) * 3);
  return sycl::range<3>(Result[0], Result[1], Result[2]);
}

// TODO: This is used by a deprecated version of
// info::kernel_device_specific::max_sub_group_size taking an input paramter.
// This should be removed when the deprecated info query is removed.
template <typename Param>
uint32_t get_kernel_device_specific_info_with_input(ur_kernel_handle_t Kernel,
                                                    ur_device_handle_t Device,
                                                    sycl::range<3>,
                                                    const AdapterPtr &Adapter) {
  static_assert(is_kernel_device_specific_info_desc<Param>::value,
                "Unexpected kernel_device_specific information descriptor");
  static_assert(std::is_same<typename Param::return_type, uint32_t>::value,
                "Unexpected return type");
  static_assert(IsSubGroupInfo<Param>::value,
                "Unexpected kernel_device_specific information descriptor for "
                "query with input");

  uint32_t Result = 0;
  // TODO catch an exception and put it to list of asynchronous exceptions
  Adapter->call<UrApiKind::urKernelGetSubGroupInfo>(
      Kernel, Device, UrInfoCode<Param>::value, sizeof(uint32_t), &Result,
      nullptr);

  return Result;
}

template <>
inline ext::intel::info::kernel_device_specific::spill_memory_size::return_type
get_kernel_device_specific_info<
    ext::intel::info::kernel_device_specific::spill_memory_size>(
    ur_kernel_handle_t Kernel, ur_device_handle_t Device,
    const AdapterPtr &Adapter) {
  size_t ResultSize = 0;

  // First call to get the number of device images
  Adapter->call<UrApiKind::urKernelGetInfo>(
      Kernel, UR_KERNEL_INFO_SPILL_MEM_SIZE, 0, nullptr, &ResultSize);

  size_t DeviceCount = ResultSize / sizeof(uint32_t);

  // Second call to retrieve the data
  std::vector<uint32_t> Device2SpillMap(DeviceCount);
  Adapter->call<UrApiKind::urKernelGetInfo>(
      Kernel, UR_KERNEL_INFO_SPILL_MEM_SIZE, ResultSize, Device2SpillMap.data(),
      nullptr);

  ur_program_handle_t Program;
  Adapter->call<UrApiKind::urKernelGetInfo>(Kernel, UR_KERNEL_INFO_PROGRAM,
                                            sizeof(ur_program_handle_t),
                                            &Program, nullptr);
  // Retrieve the associated device list
  size_t URDevicesSize = 0;
  Adapter->call<UrApiKind::urProgramGetInfo>(Program, UR_PROGRAM_INFO_DEVICES,
                                             0, nullptr, &URDevicesSize);

  std::vector<ur_device_handle_t> URDevices(URDevicesSize /
                                            sizeof(ur_device_handle_t));
  Adapter->call<UrApiKind::urProgramGetInfo>(Program, UR_PROGRAM_INFO_DEVICES,
                                             URDevicesSize, URDevices.data(),
                                             nullptr);
  assert(Device2SpillMap.size() == URDevices.size());

  // Map the result back to the program devices. UR provides the following
  // guarantee:
  //   The order of the devices is guaranteed (i.e., the same as queried by
  //   urDeviceGet) by the UR within a single application even if the runtime is
  //   reinitialized.
  for (size_t idx = 0; idx < URDevices.size(); ++idx) {
    if (URDevices[idx] == Device)
      return size_t{Device2SpillMap[idx]};
  }
  throw exception(
      make_error_code(errc::runtime),
      "ext::intel::info::kernel::spill_memory_size failed to retrieve "
      "the requested value");
}

} // namespace detail
} // namespace _V1
} // namespace sycl
