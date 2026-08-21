//==-------- cross_context_proxy.hpp --- Cross-context dependencies --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/event_impl.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
class context_impl;

/// Resolves a dependency that crosses a context boundary without holding the
/// consuming command back in the SYCL runtime.
///
/// An adapter can only express a dependency between commands of the same
/// context, so a cross-context dependency is normally connected with an empty
/// host task: the consuming command stays blocked in the graph until the
/// producing event has retired. That hides the command from the adapter, which
/// in turn cannot batch, record or otherwise pre-process it.
///
/// This is the alternative. An unsignalled, host-signalled event
/// (urEventCreateHostSignalExp) is created in the consumer's context and handed
/// to the consuming command as an ordinary UR dependency, so the command goes
/// to the adapter right away. Waiting for the producing event is left to the
/// host task thread pool, used exactly the way a host task uses it: a job there
/// blocks until the dependency has retired and then signals the proxy.
///
/// Reusing that pool is also what keeps the mechanism shutdown-safe. Every
/// proxy that has been created must eventually be signalled - a command waiting
/// on an abandoned one would keep a device submission channel blocked - and
/// draining the pool, which the runtime already does before it tears the
/// scheduler down, is what guarantees it. A host task in flight is treated
/// exactly the same way, with the same caveat: the drain is skipped on Windows,
/// where the runtime cannot wait for its own threads while the library is being
/// unloaded.
class CrossContextProxy {
public:
  /// Creates an unsignalled proxy event in \p TargetContext standing in for
  /// \p DepEvent. It stays unsignalled until the pair is handed to
  /// signalWhenRetired().
  ///
  /// \returns the proxy event, or nullptr if the dependency cannot be
  ///          represented this way (the target context does not support
  ///          host-signalled events or the producing event cannot be waited
  ///          for). The caller then has to fall back to connecting the two
  ///          contexts with a host task.
  static EventImplPtr create(context_impl &TargetContext,
                             const EventImplPtr &DepEvent);

  /// Submits a job to the host task thread pool that waits for \p DepEvent and
  /// then signals the \p Proxy created for it.
  ///
  /// The wait is a blocking one, so \p DepEvent has to be enqueued already -
  /// otherwise the pool thread would sit on a dependency that nothing is going
  /// to submit. The job signals the proxy even if the wait itself failed: the
  /// command waiting on the proxy is in the adapter's hands by then, so leaving
  /// the proxy unsignalled would strand it. A failure is instead reported to
  /// the async handler of the queue \p ConsumerEvent was submitted to, which is
  /// where the host task connection this replaces would have reported it.
  static void signalWhenRetired(const EventImplPtr &DepEvent,
                                const EventImplPtr &Proxy,
                                const EventImplPtr &ConsumerEvent);
};

} // namespace detail
} // namespace _V1
} // namespace sycl
