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
#include <sycl/queue.hpp> // for queue
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
  return_type static get(sycl::queue &queue) {
    return queue.get_device()
        .get_info<ext::oneapi::experimental::info::
                      work_group_progress_capabilities<CoordinationScope>>();
  }
};

template <execution_scope CoordinationScope>
struct sub_group_progress_capabilities {
  using return_type =
      std::vector<ext::oneapi::experimental::forward_progress_guarantee>;
  return_type static get(sycl::queue &queue) {
    return queue.get_device()
        .get_info<ext::oneapi::experimental::info::
                      sub_group_progress_capabilities<CoordinationScope>>();
  }
};

template <execution_scope CoordinationScope>
struct work_item_progress_capabilities {
  using return_type =
      std::vector<ext::oneapi::experimental::forward_progress_guarantee>;
  return_type static get(sycl::queue &queue) {
    return queue.get_device()
        .get_info<ext::oneapi::experimental::info::
                      work_item_progress_capabilities<CoordinationScope>>();
  }
};

template <>
struct PiInfoCode<work_group_progress_capabilities<
    ext::oneapi::experimental::execution_scope::root_group>> {
  static constexpr pi_device_info value = static_cast<pi_device_info>(PiCode);
};
template <>
struct is_kernel_queue_specific_info_desc<work_group_progress_capabilities<
    ext::oneapi::experimental::execution_scope::root_group>>
    : std::true_type {
  using return_type = work_group_progress_capabilities<
    ext::oneapi::experimental::execution_scope::root_group>::return_type;
};

template <>
struct PiInfoCode<sub_group_progress_capabilities<
    ext::oneapi::experimental::execution_scope::root_group>> {
  static constexpr pi_device_info value = static_cast<pi_device_info>(PiCode);
};
template <>
struct is_kernel_queue_specific_info_desc<sub_group_progress_capabilities<
    ext::oneapi::experimental::execution_scope::root_group>>
    : std::true_type {
  using return_type = sub_group_progress_capabilities<
    ext::oneapi::experimental::execution_scope::root_group>::return_type;
};


template <>
struct PiInfoCode<sub_group_progress_capabilities<
    ext::oneapi::experimental::execution_scope::work_group>> {
  static constexpr pi_device_info value = static_cast<pi_device_info>(PiCode);
};
template <>
struct is_kernel_queue_specific_info_desc<sub_group_progress_capabilities<
    ext::oneapi::experimental::execution_scope::work_group>>
    : std::true_type {
  using return_type = sub_group_progress_capabilities<
    ext::oneapi::experimental::execution_scope::work_group>::return_type;
};


template <>
struct PiInfoCode<work_item_progress_capabilities<
    ext::oneapi::experimental::execution_scope::root_group>> {
  static constexpr pi_device_info value = static_cast<pi_device_info>(PiCode);
};
template <>
struct is_kernel_queue_specific_info_desc<work_item_progress_capabilities<
    ext::oneapi::experimental::execution_scope::root_group>>
    : std::true_type {
  using return_type = work_item_progress_capabilities<
    ext::oneapi::experimental::execution_scope::root_group>::return_type;
};

template <>
struct PiInfoCode<work_item_progress_capabilities<
    ext::oneapi::experimental::execution_scope::work_group>> {
  static constexpr pi_device_info value = static_cast<pi_device_info>(PiCode);
};
template <>
struct is_kernel_queue_specific_info_desc<work_item_progress_capabilities<
    ext::oneapi::experimental::execution_scope::work_group>>
    : std::true_type {
  using return_type = work_group_progress_capabilities<
    ext::oneapi::experimental::execution_scope::root_group>::return_type;
};


template <>
struct PiInfoCode<work_item_progress_capabilities<
    ext::oneapi::experimental::execution_scope::sub_group>> {
  static constexpr pi_device_info value = static_cast<pi_device_info>(PiCode);
};
template <>
struct is_kernel_queue_specific_info_desc<work_item_progress_capabilities<
    ext::oneapi::experimental::execution_scope::sub_group>>
    : std::true_type {
  using return_type = work_item_progress_capabilities<
    ext::oneapi::experimental::execution_scope::sub_group>::return_type;
};


} // namespace kernel
} // namespace info
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
