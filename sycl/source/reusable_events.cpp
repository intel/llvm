//==------- reusable_events.cpp --- SYCL reusable events -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "detail/context_impl.hpp"
#include "detail/event_impl.hpp"
#include "detail/queue_impl.hpp"
#include <sycl/detail/ur.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

namespace detail {

__SYCL_EXPORT sycl::event make_event(const sycl::context &ctxt,
                                     uint32_t Flags) {
  const bool EnableProfiling = Flags & make_event_flag_enable_profiling;
  const bool EnableIPC = Flags & make_event_flag_enable_ipc;

  // enable_profiling and enable_ipc are mutually exclusive.
  if (EnableProfiling && EnableIPC) {
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "The enable_profiling and enable_ipc properties cannot both be set "
        "when creating an event.");
  }

  detail::context_impl &ContextImpl = *sycl::detail::getSyclObjImpl(ctxt);

  if (EnableProfiling && !ContextImpl.supportsEventProfiling()) {
    throw sycl::exception(sycl::make_error_code(errc::feature_not_supported),
                          "Context does not support per-event profiling.");
  }

  // enable_ipc requires every device in the context to support IPC events.
  if (EnableIPC && !ContextImpl.supportsIPCEvents()) {
    throw sycl::exception(sycl::make_error_code(errc::feature_not_supported),
                          "Not all devices in the context support "
                          "aspect::ext_oneapi_ipc_event.");
  }

  sycl::event RetEvent{};
  detail::event_impl &EventImpl = *sycl::detail::getSyclObjImpl(RetEvent);
  EventImpl.setContextImpl(ContextImpl);
  EventImpl.setProfilingEnabled(EnableProfiling);
  EventImpl.setIPCEnabled(EnableIPC);

  // The backend UR event is created lazily on first signal or first
  // ipc::event::get.
  return RetEvent;
}

} // namespace detail

__SYCL_EXPORT void enqueue_wait_event(sycl::queue q, const event &evt) {
  detail::queue_impl &QueueImpl = *sycl::detail::getSyclObjImpl(q);

  QueueImpl.submit_barrier_direct_without_event(
      sycl::span<const event>(&evt, 1), detail::CGType::BarrierWaitlist,
      detail::code_location::current());
}

__SYCL_EXPORT void enqueue_wait_events(sycl::queue q,
                                       const std::vector<event> &evts) {
  detail::queue_impl &QueueImpl = *sycl::detail::getSyclObjImpl(q);

  QueueImpl.submit_barrier_direct_without_event(
      evts, detail::CGType::BarrierWaitlist, detail::code_location::current());
}

__SYCL_EXPORT void enqueue_signal_event(sycl::queue q, event &evt) {
  detail::queue_impl &QueueImpl = *sycl::detail::getSyclObjImpl(q);
  detail::event_impl &EventImpl = *sycl::detail::getSyclObjImpl(evt);

  if (EventImpl.isInterop()) {
    throw sycl::exception(
        sycl::make_error_code(errc::runtime),
        "Enqueueing an interop event for signaling is not supported.");
  }

  if (QueueImpl.hasCommandGraph()) {
    throw sycl::exception(sycl::make_error_code(errc::runtime),
                          "Enqueueing an event for signaling is not supported "
                          "on a queue which is recording a graph.");
  }

  detail::context_impl &QueueContextImpl =
      *sycl::detail::getSyclObjImpl(q.get_context());

  detail::context_impl &EventContextImpl = EventImpl.getContextImpl();

  if (&QueueContextImpl != &EventContextImpl) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "Event context must match the queue context.");
  }

  // An IPC event cannot be signaled on a profiling-enabled queue.
  if (EventImpl.isIPCEnabled() && QueueImpl.MIsProfilingEnabled) {
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "An IPC-enabled event cannot be signaled on a queue that has "
        "profiling enabled.");
  }

  QueueImpl.submit_barrier_direct_without_event(
      {}, detail::CGType::Barrier, detail::code_location::current(),
      sycl::detail::getSyclObjImpl(evt));
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
