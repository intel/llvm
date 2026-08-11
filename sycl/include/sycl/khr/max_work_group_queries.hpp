//==-- max_work_group_queries.hpp - oneapi max work groups info traits
//------------==//
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
namespace khr::info::device {

template <int Dimensions = 3> struct max_work_group_range;

// max_work_group_range<1> and <2> are RT-only; only <3> dispatches via UR.
template <>
struct max_work_group_range<1>
    : sycl::detail::rt_traits_base<sycl::detail::info_class::device> {
  using return_type = sycl::range<1>;
};

template <>
struct max_work_group_range<2>
    : sycl::detail::rt_traits_base<sycl::detail::info_class::device> {
  using return_type = sycl::range<2>;
};

template <>
struct max_work_group_range<3>
    : sycl::detail::ur_traits_base<sycl::detail::info_class::device,
                                   UR_DEVICE_INFO_MAX_WORK_GROUPS_3D> {
  using return_type = sycl::range<3>;
};

struct max_work_group_range_size
    : sycl::detail::ur_traits_base<sycl::detail::info_class::device,
                                   UR_DEVICE_INFO_MAX_WORK_GROUPS> {
  using return_type = size_t;
};

} // namespace khr::info::device
} // namespace _V1
} // namespace sycl
