//==---------------- event_binding.hpp - SYCL event binding ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

class event_impl;

/// The state of one signal of an event.
///
/// An event may be enqueued for signaling more than once (see
/// sycl_ext_oneapi_reusable_events). Each such enqueue is a separate piece of
/// work with dependencies of its own, so the state which belongs to one signal
/// rather than to the event is kept here. An event_impl points at its current
/// binding, the command which produces the signal owns the binding it writes
/// to, and both are the same object as long as the event is not enqueued for
/// signaling again.
///
/// The binding is created by the thread which owns the event, before the
/// event is visible to other threads.
class event_binding {
public:
  /// Dependency events prepared for waiting by backend.
  /// See Command::processDepEvent for details.
  std::vector<std::shared_ptr<event_impl>> MPreparedDepsEvents;
  std::vector<std::shared_ptr<event_impl>> MPreparedHostDepsEvents;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
