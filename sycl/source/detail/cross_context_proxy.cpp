//==-------- cross_context_proxy.cpp --- Cross-context dependencies --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/adapter_impl.hpp>
#include <detail/context_impl.hpp>
#include <detail/cross_context_proxy.hpp>
#include <detail/global_handler.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/thread_pool.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

EventImplPtr CrossContextProxy::create(context_impl &TargetContext,
                                       const EventImplPtr &DepEvent) {
  // A discarded event never gets a UR handle, so it cannot be waited for.
  // Nothing can be done for a dependency without a context either.
  if (DepEvent->isDiscarded() || DepEvent->isHost())
    return nullptr;

  if (!TargetContext.supportsHostSignalEvents())
    return nullptr;

  ur_event_handle_t Handle = nullptr;
  adapter_impl &Adapter = TargetContext.getAdapter();
  const ur_result_t Res =
      Adapter.call_nocheck<UrApiKind::urEventCreateHostSignalExp>(
          TargetContext.getHandleRef(), &Handle);
  if (Res != UR_RESULT_SUCCESS || !Handle)
    return nullptr;

  // The proxy owns the UR event from now on; event_impl releases it. The
  // interop path is deliberately not used: it would query the event's context
  // back from the adapter just to compare it against the context the event was
  // just created in, and it would mark the event complete on construction.
  return event_impl::create_from_owned_handle(Handle, TargetContext);
}

void CrossContextProxy::signalWhenRetired(const EventImplPtr &DepEvent,
                                          const EventImplPtr &Proxy,
                                          const EventImplPtr &ConsumerEvent) {
  // The pool that runs host tasks, used the same way: the job occupies a thread
  // until the dependency it waits for has retired.
  GlobalHandler::instance().getHostTaskThreadPool().submit(
      [Dep = DepEvent, Proxy = Proxy, Consumer = ConsumerEvent]() {
        // Nothing can be failed synchronously from here - the consuming command
        // has been submitted already - so a failure goes to the async handler
        // of its queue, which is what a host task does when waiting for its
        // dependencies fails.
        auto Report = [&Consumer]() {
          if (std::shared_ptr<queue_impl> Queue = Consumer->getSubmittedQueue())
            Scheduler::getInstance().reportAsyncException(
                Queue, std::current_exception());
        };

        try {
          // Waiting, rather than querying the completion status, is also what
          // makes the producing device's writes visible to whoever comes next.
          Dep->waitInternal();
        } catch (...) {
          Report();
        }

        try {
          // Signalled even if the wait above failed: a command waiting on an
          // unsignalled proxy would never start.
          Proxy->getAdapter().call<UrApiKind::urEventHostSignalExp>(
              Proxy->getHandle());
        } catch (...) {
          Report();
        }
      });
}

} // namespace detail
} // namespace _V1
} // namespace sycl
