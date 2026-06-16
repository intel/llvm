//==----- event.hpp - SYCL event information descriptors -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/info_desc_traits.hpp>
#include <unified-runtime/ur_api.h>

#include <cstdint>

namespace sycl {
inline namespace _V1 {
namespace info {

// A.6 Event information desctiptors
enum class event_command_status : int32_t {
  submitted = UR_EVENT_STATUS_SUBMITTED,
  running = UR_EVENT_STATUS_RUNNING,
  complete = UR_EVENT_STATUS_COMPLETE,
  // Since all BE values are positive, it is safe to use a negative value If you
  // add other ext_oneapi values
  ext_oneapi_unknown = -1
};

namespace event {
template <ur_event_info_t UrCode>
using event_traits =
    sycl::detail::ur_traits_base<sycl::detail::info_class::event, UrCode>;

struct command_execution_status
    : event_traits<UR_EVENT_INFO_COMMAND_EXECUTION_STATUS> {
  using return_type = info::event_command_status;
};
struct reference_count : event_traits<UR_EVENT_INFO_REFERENCE_COUNT> {
  using return_type = uint32_t;
};
} // namespace event
namespace event_profiling {
template <ur_profiling_info_t UrCode>
using profiling_traits =
    sycl::detail::ur_traits_base<sycl::detail::info_class::event_profiling,
                                 UrCode>;

struct command_submit : profiling_traits<UR_PROFILING_INFO_COMMAND_SUBMIT> {
  using return_type = uint64_t;
};
struct command_start : profiling_traits<UR_PROFILING_INFO_COMMAND_START> {
  using return_type = uint64_t;
};
struct command_end : profiling_traits<UR_PROFILING_INFO_COMMAND_END> {
  using return_type = uint64_t;
};
} // namespace event_profiling

} // namespace info

namespace detail {
// SFINAE predicates confining `event::get_info<T>()` to event traits and
// `event::get_profiling_info<T>()` to event_profiling traits. The
// `return_type` alias is load-bearing for ABI symbol mangling — keep stable.
template <typename T>
struct is_event_info_desc : is_info_desc_for<T, info_class::event> {};
template <typename T>
struct is_event_profiling_info_desc
    : is_info_desc_for<T, info_class::event_profiling> {};
} // namespace detail
} // namespace _V1
} // namespace sycl
