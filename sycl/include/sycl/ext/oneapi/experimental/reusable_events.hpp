//==------- reusable_events.hpp --- SYCL reusable events -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/context.hpp>
#include <sycl/device.hpp>
#include <sycl/event.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/platform.hpp>
#include <sycl/queue.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

struct enable_profiling_key
    : detail::compile_time_property_key<detail::PropKind::EnableProfiling> {
  using value_t = property_value<enable_profiling_key>;
};

inline constexpr enable_profiling_key::value_t enable_profiling;

namespace detail {

enum make_event_flags : uint32_t { make_event_flag_enable_profiling = 1u };

__SYCL_EXPORT sycl::event make_event(const sycl::context &ctxt, uint32_t Flags);

template <typename PropertyListT> uint32_t getMakeEventFlags() {
  uint32_t Flags = 0;
  if constexpr (PropertyListT::template has_property<enable_profiling_key>())
    Flags |= make_event_flag_enable_profiling;
  return Flags;
}
} // namespace detail

template <typename PropertyListT = empty_properties_t>
inline sycl::event make_event(const sycl::context &ctxt,
                              PropertyListT props = {}) {
  static_assert(is_property_list_v<PropertyListT>,
                "Props must be a sycl::ext::oneapi::experimental::properties");
  (void)props;

  return detail::make_event(ctxt, detail::getMakeEventFlags<PropertyListT>());
}

template <typename PropertyListT = empty_properties_t>
inline sycl::event make_event(PropertyListT props = {}) {
  sycl::device Dev;
  sycl::context Ctx = Dev.get_platform().khr_get_default_context();
  return make_event(Ctx, props);
}

__SYCL_EXPORT void enqueue_wait_event(sycl::queue q, const event &evt);
__SYCL_EXPORT void enqueue_wait_events(sycl::queue q,
                                       const std::vector<event> &evts);
__SYCL_EXPORT void enqueue_signal_event(sycl::queue q, event &evt);

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
