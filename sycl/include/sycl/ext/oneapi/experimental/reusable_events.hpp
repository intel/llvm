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
#include <sycl/ext/oneapi/experimental/detail/ipc_common.hpp> // enable_ipc_key
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/platform.hpp>
#include <sycl/queue.hpp>

#include <cstdint>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

struct enable_profiling_key
    : detail::compile_time_property_key<detail::PropKind::EnableProfiling> {
  using value_t = property_value<enable_profiling_key>;
};

inline constexpr enable_profiling_key::value_t enable_profiling;

// enable_ipc (from sycl_ext_oneapi_inter_process_communication, defined in
// detail/ipc_common.hpp) is a valid property for make_event too.
template <>
struct is_property_key_of<enable_ipc_key, sycl::event> : std::true_type {};

namespace detail {
// Implementation-internal flags describing an event to create. They are
// translated to UR event flags in the runtime library. Kept as a bitfield
// (rather than separate bools) so make_event can grow more properties without
// changing this ABI-exported signature. See getMakeEventFlags below.
enum make_event_flags : uint32_t {
  make_event_flag_enable_profiling = 1u << 0,
  make_event_flag_enable_ipc = 1u << 1,
};

// Non-templated implementation entry point defined in the SYCL runtime
// library. Flags is a bitwise OR of make_event_flags values.
__SYCL_EXPORT sycl::event make_event(const sycl::context &ctxt, uint32_t Flags);

// Lowers a compile-time property list to the runtime flags word.
template <typename PropertyListT> uint32_t getMakeEventFlags() {
  uint32_t Flags = 0;
  if constexpr (PropertyListT::template has_property<enable_profiling_key>())
    Flags |= make_event_flag_enable_profiling;
  if constexpr (PropertyListT::template has_property<enable_ipc_key>())
    Flags |= make_event_flag_enable_ipc;
  return Flags;
}
} // namespace detail

template <typename PropertyListT = empty_properties_t>
sycl::event make_event(const sycl::context &ctxt, PropertyListT props = {}) {
  static_assert(is_property_list_v<PropertyListT>,
                "Props must be a sycl::ext::oneapi::experimental::properties");
  (void)props;
  return detail::make_event(ctxt,
                            detail::getMakeEventFlags<PropertyListT>());
}

template <typename PropertyListT = empty_properties_t>
sycl::event make_event(PropertyListT props = {}) {
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
