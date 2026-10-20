//==----- queue.hpp - SYCL queue information descriptors -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/info_desc_traits.hpp>
#include <unified-runtime/ur_api.h>

#include <cstdint>

namespace sycl {
inline namespace _V1 {

class context;
class device;

namespace info {
// A.4 Queue information descriptors
namespace queue {
template <ur_queue_info_t UrCode>
using queue_traits =
    sycl::detail::ur_traits_base<sycl::detail::info_class::queue, UrCode>;

struct context : queue_traits<UR_QUEUE_INFO_CONTEXT> {
  using return_type = sycl::context;
};
struct device : queue_traits<UR_QUEUE_INFO_DEVICE> {
  using return_type = sycl::device;
};
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
struct __SYCL_DEPRECATED("info::queue::reference_count is not part of "
                         "SYCL 2020") reference_count
    : queue_traits<UR_QUEUE_INFO_REFERENCE_COUNT> {
  using return_type = uint32_t;
};
#endif // __INTEL_PREVIEW_BREAKING_CHANGES
} // namespace queue
} // namespace info

namespace detail {
// SFINAE predicate confining `queue::get_info<T>()` to queue traits.
// `return_type` alias is load-bearing for ABI symbol mangling — keep stable.
template <typename T>
struct is_queue_info_desc : is_info_desc_for<T, info_class::queue> {};
} // namespace detail
} // namespace _V1
} // namespace sycl
