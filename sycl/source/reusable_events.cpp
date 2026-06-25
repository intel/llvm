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
#include <detail/sycl_mem_obj_t.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

namespace detail {

__SYCL_EXPORT sycl::event make_event(const sycl::context &ctxt,
                                     bool enable_profiling) {
  const auto &ContextDevices = ctxt.get_devices();
  detail::context_impl &ContextImpl = *sycl::detail::getSyclObjImpl(ctxt);
  sycl::detail::adapter_impl &Adapter = ContextImpl.getAdapter();

  // TODO reusable events - check platform support for per-event profiling

  if (ContextDevices.size() == 0) {
    throw sycl::exception(sycl::make_error_code(errc::runtime),
                          "Context needs to have at least one device.");
  }

  // TODO reusable events
  if (!ContextImpl.supportsReusableEvents()) {
    throw sycl::exception(sycl::make_error_code(errc::runtime),
                          "Not implemented yet.");
  }

  ur_event_handle_t EventHandle = nullptr;
  ur_exp_event_desc_t Desc = {};
  Desc.stype = UR_STRUCTURE_TYPE_EXP_EVENT_DESC;
  Desc.hDevice = detail::getSyclObjImpl(ContextDevices[0])->getHandleRef();
  if (enable_profiling) {
    Desc.flags = UR_EXP_EVENT_FLAG_ENABLE_PROFILING;
  }

  ur_result_t Result =
      Adapter.call_nocheck<sycl::detail::UrApiKind::urEventCreateExp>(
          ContextImpl.getHandleRef(), &Desc, &EventHandle);
  if (Result != UR_RESULT_SUCCESS) {
    throw sycl::exception(sycl::make_error_code(errc::runtime),
                          "Failed to create an event.");
  }

  auto ResEvent = detail::event_impl::create_from_handle(EventHandle, ctxt);
  ResEvent->setStateIncomplete();
  ResEvent->setProfilingEnabled(enable_profiling);

  return detail::createSyclObjFromImpl<sycl::event>(std::move(ResEvent));
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

  if (QueueImpl.hasCommandGraph()) {
    throw sycl::exception(sycl::make_error_code(errc::runtime),
                          "Enqueueing an event for signaling is not supported "
                          "on a queue which is recording a graph.");
  }

  detail::context_impl &ContextImpl =
      *sycl::detail::getSyclObjImpl(q.get_context());

  // TODO reusable events
  if (!ContextImpl.supportsReusableEvents()) {
    throw sycl::exception(sycl::make_error_code(errc::runtime),
                          "Not implemented yet.");
  }

  detail::context_impl &EventContextImpl =
      sycl::detail::getSyclObjImpl(evt)->getContextImpl();

  if (&ContextImpl != &EventContextImpl) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "Event context must match the queue context.");
  }

  QueueImpl.submit_barrier_direct_without_event(
      {}, detail::CGType::Barrier, detail::code_location::current(),
      detail::getSyclObjImpl(evt));
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
