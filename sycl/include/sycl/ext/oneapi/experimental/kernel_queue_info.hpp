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

using kqs_traits =
    sycl::detail::rt_traits_base<sycl::detail::info_class::kernel_queue_specific>;

struct max_num_work_groups : kqs_traits {
  using return_type = size_t;
};
struct max_work_group_size : kqs_traits {
  using return_type = size_t;
};
struct max_sub_group_size : kqs_traits {
  using return_type = uint32_t;
};
struct num_sub_groups : kqs_traits {
  using return_type = uint32_t;
};

template <int Dim> struct max_work_item_sizes : kqs_traits {
  using return_type = sycl::id<Dim>;
};

} // namespace ext::oneapi::experimental::info::kernel_queue_specific
} // namespace _V1
} // namespace sycl
