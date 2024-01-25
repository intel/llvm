//==---------------- event_info.hpp - SYCL event ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/event_impl.hpp>
#include <detail/plugin.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/info_desc_helpers.hpp>
#include <sycl/info/info_desc.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

template <typename Param>
typename Param::return_type
get_event_profiling_info(sycl::detail::pi::PiEvent Event,
                         const PluginPtr &Plugin) {
  static_assert(is_event_profiling_info_desc<Param>::value,
                "Unexpected event profiling info descriptor");
  typename Param::return_type Result{0};
  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin->call<PiApiKind::piEventGetProfilingInfo>(
      Event, PiInfoCode<Param>::value, sizeof(Result), &Result, nullptr);
  return Result;
}

template <typename Param>
typename Param::return_type
get_event_profiling_info(sycl::detail::pi::PiEvent Event,
                         sycl::detail::pi::PiExtSyncPoint SyncPoint,
                         const PluginPtr &Plugin) {
  static_assert(is_event_profiling_info_desc<Param>::value,
                "Unexpected event profiling info descriptor");
  typename Param::return_type Result{0};
  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin->call<PiApiKind::piextSyncPointGetProfilingInfo>(
      Event, SyncPoint, PiInfoCode<Param>::value, sizeof(Result), &Result,
      nullptr);
  return Result;
}

template <typename Param>
typename Param::return_type get_event_info(sycl::detail::pi::PiEvent Event,
                                           const PluginPtr &Plugin) {
  static_assert(is_event_info_desc<Param>::value,
                "Unexpected event info descriptor");
  typename Param::return_type Result{0};
  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin->call<PiApiKind::piEventGetInfo>(Event, PiInfoCode<Param>::value,
                                          sizeof(Result), &Result, nullptr);
  return Result;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
