//==----------- kernel_launch_queries.hpp - sycl_ext_oneapi_launch_queries
//------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/info_desc_helpers.hpp> // for is_kernel_queue_specific_info_desc
#include <sycl/ext/oneapi/experimental/forward_progress.hpp> // for forward_progress_guarantee and execution_scope
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {
namespace info {
namespace kernel {

template <execution_scope CoordinationScope>
struct work_group_progress_capabilities {
  using return_type =
      std::vector<ext::oneapi::experimental::forward_progress_guarantee>;
  using device_desc =
      ext::oneapi::experimental::info::device::work_group_progress_capabilities<
          CoordinationScope>;
};

template <execution_scope CoordinationScope>
struct sub_group_progress_capabilities {
  using return_type =
      std::vector<ext::oneapi::experimental::forward_progress_guarantee>;
  using device_desc =
      ext::oneapi::experimental::info::device::sub_group_progress_capabilities<
          CoordinationScope>;
};

template <execution_scope CoordinationScope>
struct work_item_progress_capabilities {
  using return_type =
      std::vector<ext::oneapi::experimental::forward_progress_guarantee>;
  using device_desc =
      ext::oneapi::experimental::info::device::work_item_progress_capabilities<
          CoordinationScope>;
};
} // namespace kernel
} // namespace info
} // namespace experimental
} // namespace oneapi
} // namespace ext

namespace detail {

template <>
struct PiInfoCode<
    ext::oneapi::experimental::info::kernel::work_group_progress_capabilities<
        ext::oneapi::experimental::execution_scope::root_group>> {
  static constexpr pi_device_info value = static_cast<pi_device_info>(
      PI_EXT_ONEAPI_KERNEL_INFO_WORK_GROUP_PROGRESS_AT_ROOT_GROUP_LEVEL);
};
template <>
struct is_kernel_device_specific_info_desc<
    ext::oneapi::experimental::info::kernel::work_group_progress_capabilities<
        ext::oneapi::experimental::execution_scope::root_group>>
    : std::true_type {
  using return_type =
      ext::oneapi::experimental::info::kernel::work_group_progress_capabilities<
          ext::oneapi::experimental::execution_scope::root_group>::return_type;
};

template <>
struct PiInfoCode<
    ext::oneapi::experimental::info::kernel::sub_group_progress_capabilities<
        ext::oneapi::experimental::execution_scope::root_group>> {
  static constexpr pi_device_info value = static_cast<pi_device_info>(
      PI_EXT_ONEAPI_KERNEL_INFO_SUB_GROUP_PROGRESS_AT_ROOT_GROUP_LEVEL);
};
template <>
struct is_kernel_device_specific_info_desc<
    ext::oneapi::experimental::info::kernel::sub_group_progress_capabilities<
        ext::oneapi::experimental::execution_scope::root_group>>
    : std::true_type {
  using return_type =
      ext::oneapi::experimental::info::kernel::sub_group_progress_capabilities<
          ext::oneapi::experimental::execution_scope::root_group>::return_type;
};

template <>
struct PiInfoCode<
    ext::oneapi::experimental::info::kernel::sub_group_progress_capabilities<
        ext::oneapi::experimental::execution_scope::work_group>> {
  static constexpr pi_device_info value = static_cast<pi_device_info>(
      PI_EXT_ONEAPI_KERNEL_INFO_SUB_GROUP_PROGRESS_AT_WORK_GROUP_LEVEL);
};
template <>
struct is_kernel_device_specific_info_desc<
    ext::oneapi::experimental::info::kernel::sub_group_progress_capabilities<
        ext::oneapi::experimental::execution_scope::work_group>>
    : std::true_type {
  using return_type =
      ext::oneapi::experimental::info::kernel::sub_group_progress_capabilities<
          ext::oneapi::experimental::execution_scope::work_group>::return_type;
};

template <>
struct PiInfoCode<
    ext::oneapi::experimental::info::kernel::work_item_progress_capabilities<
        ext::oneapi::experimental::execution_scope::root_group>> {
  static constexpr pi_device_info value = static_cast<pi_device_info>(
      PI_EXT_ONEAPI_KERNEL_INFO_WORK_ITEM_PROGRESS_AT_ROOT_GROUP_LEVEL);
};
template <>
struct is_kernel_device_specific_info_desc<
    ext::oneapi::experimental::info::kernel::work_item_progress_capabilities<
        ext::oneapi::experimental::execution_scope::root_group>>
    : std::true_type {
  using return_type =
      ext::oneapi::experimental::info::kernel::work_item_progress_capabilities<
          ext::oneapi::experimental::execution_scope::root_group>::return_type;
};

template <>
struct PiInfoCode<
    ext::oneapi::experimental::info::kernel::work_item_progress_capabilities<
        ext::oneapi::experimental::execution_scope::work_group>> {
  static constexpr pi_device_info value = static_cast<pi_device_info>(
      PI_EXT_ONEAPI_KERNEL_INFO_WORK_ITEM_PROGRESS_AT_WORK_GROUP_LEVEL);
};
template <>
struct is_kernel_device_specific_info_desc<
    ext::oneapi::experimental::info::kernel::work_item_progress_capabilities<
        ext::oneapi::experimental::execution_scope::work_group>>
    : std::true_type {
  using return_type =
      ext::oneapi::experimental::info::kernel::work_group_progress_capabilities<
          ext::oneapi::experimental::execution_scope::root_group>::return_type;
};

template <>
struct PiInfoCode<
    ext::oneapi::experimental::info::kernel::work_item_progress_capabilities<
        ext::oneapi::experimental::execution_scope::sub_group>> {
  static constexpr pi_device_info value = static_cast<pi_device_info>(
      PI_EXT_ONEAPI_KERNEL_INFO_WORK_ITEM_PROGRESS_AT_SUB_GROUP_LEVEL);
};
template <>
struct is_kernel_device_specific_info_desc<
    ext::oneapi::experimental::info::kernel::work_item_progress_capabilities<
        ext::oneapi::experimental::execution_scope::sub_group>>
    : std::true_type {
  using return_type =
      ext::oneapi::experimental::info::kernel::work_item_progress_capabilities<
          ext::oneapi::experimental::execution_scope::sub_group>::return_type;
};
} // namespace detail
} // namespace _V1
} // namespace sycl
