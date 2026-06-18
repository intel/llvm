//==------- reusable_events.hpp --- SYCL reusable events -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/event.hpp>
#include <sycl/device.hpp>
#include <sycl/context.hpp>
#include <sycl/queue.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

template <typename PropertyListT = empty_properties_t>
sycl::event make_event(const sycl::context &ctxt, PropertyListT props = {});

template <typename PropertyListT = empty_properties_t>
sycl::event make_event(PropertyListT props = {}) {
  sycl::device dev;
  sycl::context Ctx = dev.get_platform().khr_get_default_context();
  return make_event(dev, Ctx, Props);
}

void enqueue_wait_event(sycl::queue q, const event& evt);
void enqueue_wait_events(sycl::queue q, const std::vector<event>& evts);
void enqueue_signal_event(sycl::queue q, event& evt);

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
