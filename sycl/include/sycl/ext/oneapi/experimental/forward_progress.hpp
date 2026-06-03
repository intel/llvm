//==------- forward_progress.hpp - sycl_ext_oneapi_forward_progress -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/info_desc_traits.hpp>

#include <vector>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {

enum class forward_progress_guarantee { concurrent, parallel, weakly_parallel };

enum class execution_scope {
  work_item,
  sub_group,
  work_group,
  root_group,
};

namespace info::device {

template <execution_scope CoordinationScope>
struct work_group_progress_capabilities {
  using return_type = std::vector<forward_progress_guarantee>;
  using info_class = sycl::detail::info_class::device;
  // RT-only: dispatched via explicit CASE in device_impl.hpp; no UR enum.
};

template <execution_scope CoordinationScope>
struct sub_group_progress_capabilities {
  using return_type = std::vector<forward_progress_guarantee>;
  using info_class = sycl::detail::info_class::device;
  // RT-only: dispatched via explicit CASE in device_impl.hpp; no UR enum.
};

template <execution_scope CoordinationScope>
struct work_item_progress_capabilities {
  using return_type = std::vector<forward_progress_guarantee>;
  using info_class = sycl::detail::info_class::device;
  // RT-only: dispatched via explicit CASE in device_impl.hpp; no UR enum.
};

} // namespace info::device

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
