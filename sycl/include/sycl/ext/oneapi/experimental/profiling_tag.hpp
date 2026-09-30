//==--------- profiling_tag.hpp --- SYCL profiling tag extension -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/aspects.hpp>
#include <sycl/event.hpp>
#include <sycl/handler.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/queue.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

inline event submit_profiling_tag(queue &Queue,
                                  const sycl::detail::code_location &CodeLoc =
                                      sycl::detail::code_location::current()) {
  // The profiling tag can be serviced natively when the device advertises the
  // ext_oneapi_queue_profiling_tag aspect. Otherwise, we can still service it
  // as long as the queue has profiling enabled. In both cases we submit an
  // internal profiling-tag command group: the runtime records the timestamp
  // using a native device command where possible and otherwise falls back to a
  // barrier (see CGType::ProfilingTag handling in the scheduler).
  if (!Queue.get_device().has(aspect::ext_oneapi_queue_profiling_tag) &&
      !Queue.has_property<sycl::property::queue::enable_profiling>())
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Device must either have aspect::ext_oneapi_queue_profiling_tag or the "
        "queue must have profiling enabled.");

  return Queue.submit(
      [=](handler &CGH) {
        sycl::detail::HandlerAccess::internalProfilingTagImpl(CGH);
      },
      CodeLoc);
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
