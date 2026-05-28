//==-- max_work_groups.hpp - oneapi max work groups info traits ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/info_desc_traits.hpp>
#include <sycl/id.hpp>
#include <unified-runtime/ur_api.h>

#include <cstddef>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental::info::device {

struct max_global_work_groups {
  using return_type = size_t;
  using info_class = sycl::detail::info_class::device;
  static constexpr ur_device_info_t ur_code = UR_DEVICE_INFO_MAX_WORK_GROUPS;
};

template <int Dim> struct max_work_groups {
  using return_type = sycl::id<Dim>;
  using info_class = sycl::detail::info_class::device;
  static constexpr ur_device_info_t ur_code =
      Dim == 3 ? UR_DEVICE_INFO_MAX_WORK_GROUPS_3D : ur_device_info_t(0);
};

} // namespace ext::oneapi::experimental::info::device
} // namespace _V1
} // namespace sycl
