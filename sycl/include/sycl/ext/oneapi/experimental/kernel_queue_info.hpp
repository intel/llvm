//==-- kernel_queue_info.hpp - oneapi kernel_queue_specific info traits ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/info_desc_traits.hpp>
#include <sycl/id.hpp>

#include <cstddef>
#include <cstdint>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental::info::kernel_queue_specific {

struct max_num_work_groups {
  using return_type = size_t;
  using info_class = sycl::detail::info_class::kernel_queue_specific;
  static constexpr int ur_code = 0;
};

struct max_work_group_size {
  using return_type = size_t;
  using info_class = sycl::detail::info_class::kernel_queue_specific;
  static constexpr int ur_code = 0;
};

struct max_sub_group_size {
  using return_type = uint32_t;
  using info_class = sycl::detail::info_class::kernel_queue_specific;
  static constexpr int ur_code = 0;
};

struct num_sub_groups {
  using return_type = uint32_t;
  using info_class = sycl::detail::info_class::kernel_queue_specific;
  static constexpr int ur_code = 0;
};

template <int Dim> struct max_work_item_sizes {
  using return_type = sycl::id<Dim>;
  using info_class = sycl::detail::info_class::kernel_queue_specific;
  static constexpr int ur_code = 0;
};

} // namespace ext::oneapi::experimental::info::kernel_queue_specific
} // namespace _V1
} // namespace sycl
