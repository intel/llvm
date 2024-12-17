//==-------------- queue.cpp -----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/backend_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/event.hpp>
#include <sycl/exception_list.hpp>
#include <sycl/handler.hpp>
#include <sycl/queue.hpp>

#include <algorithm>

namespace sycl {
inline namespace _V1 {

namespace detail {
SubmissionInfo::SubmissionInfo()
    : impl{std::make_shared<SubmissionInfoImpl>()} {}

optional<SubmitPostProcessF> &SubmissionInfo::PostProcessorFunc() {
  return impl->MPostProcessorFunc;
}

const optional<SubmitPostProcessF> &SubmissionInfo::PostProcessorFunc() const {
  return impl->MPostProcessorFunc;
}

std::shared_ptr<detail::queue_impl> &SubmissionInfo::SecondaryQueue() {
  return impl->MSecondaryQueue;
}

const std::shared_ptr<detail::queue_impl> &
SubmissionInfo::SecondaryQueue() const {
  return impl->MSecondaryQueue;
}

ext::oneapi::experimental::event_mode_enum &SubmissionInfo::EventMode() {
  return impl->MEventMode;
}

const ext::oneapi::experimental::event_mode_enum &
SubmissionInfo::EventMode() const {
  return impl->MEventMode;
}
} // namespace detail

queue::queue(const context &SyclContext, const device_selector &DeviceSelector,
             const async_handler &AsyncHandler, const property_list &PropList) {
  const std::vector<device> Devs = SyclContext.get_devices();

  auto Comp = [&DeviceSelector](const device &d1, const device &d2) {
    return DeviceSelector(d1) < DeviceSelector(d2);
  };

  const device &SyclDevice = *std::max_element(Devs.begin(), Devs.end(), Comp);

  impl = std::make_shared<detail::queue_impl>(
      detail::getSyclObjImpl(SyclDevice), detail::getSyclObjImpl(SyclContext),
      AsyncHandler, PropList);
}

queue::queue(const context &SyclContext, const device &SyclDevice,
             const async_handler &AsyncHandler, const property_list &PropList) {
  impl = std::make_shared<detail::queue_impl>(
      detail::getSyclObjImpl(SyclDevice), detail::getSyclObjImpl(SyclContext),
      AsyncHandler, PropList);
}

queue::queue(const device &SyclDevice, const async_handler &AsyncHandler,
             const property_list &PropList) {
  impl = std::make_shared<detail::queue_impl>(
      detail::getSyclObjImpl(SyclDevice), AsyncHandler, PropList);
}

queue::queue(const context &SyclContext, const device_selector &deviceSelector,
             const property_list &PropList)
    : queue(SyclContext, deviceSelector,
            detail::getSyclObjImpl(SyclContext)->get_async_handler(),
            PropList) {}

queue::queue(const context &SyclContext, const device &SyclDevice,
             const property_list &PropList)
    : queue(SyclContext, SyclDevice,
            detail::getSyclObjImpl(SyclContext)->get_async_handler(),
            PropList) {}

queue::queue(cl_command_queue clQueue, const context &SyclContext,
             const async_handler &AsyncHandler) {
  const property_list PropList{};
  impl = std::make_shared<detail::queue_impl>(
      // TODO(pi2ur): Don't cast straight from cl_command_queue
      reinterpret_cast<ur_queue_handle_t>(clQueue),
      detail::getSyclObjImpl(SyclContext), AsyncHandler, PropList);
}

cl_command_queue queue::get() const { return impl->get(); }

context queue::get_context() const { return impl->get_context(); }

device queue::get_device() const { return impl->get_device(); }

ext::oneapi::experimental::queue_state queue::ext_oneapi_get_state() const {
  return impl->getCommandGraph()
             ? ext::oneapi::experimental::queue_state::recording
             : ext::oneapi::experimental::queue_state::executing;
}

ext::oneapi::experimental::command_graph<
    ext::oneapi::experimental::graph_state::modifiable>
queue::ext_oneapi_get_graph() const {
  auto Graph = impl->getCommandGraph();
  if (!Graph)
    throw sycl::exception(
        make_error_code(errc::invalid),
        "ext_oneapi_get_graph() can only be called on recording queues.");

  return sycl::detail::createSyclObjFromImpl<
      ext::oneapi::experimental::command_graph<
          ext::oneapi::experimental::graph_state::modifiable>>(Graph);
}

void queue::throw_asynchronous() { impl->throw_asynchronous(); }

event queue::memset(void *Ptr, int Value, size_t Count,
                    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->memset(impl, Ptr, Value, Count, {}, /*CallerNeedsEvent=*/true);
}

event queue::memset(void *Ptr, int Value, size_t Count, event DepEvent,
                    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->memset(impl, Ptr, Value, Count, {DepEvent},
                      /*CallerNeedsEvent=*/true);
}

event queue::memset(void *Ptr, int Value, size_t Count,
                    const std::vector<event> &DepEvents,
                    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->memset(impl, Ptr, Value, Count, DepEvents,
                      /*CallerNeedsEvent=*/true);
}

event queue::memcpy(void *Dest, const void *Src, size_t Count,
                    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->memcpy(impl, Dest, Src, Count, {}, /*CallerNeedsEvent=*/true,
                      TlsCodeLocCapture.query());
}

event queue::memcpy(void *Dest, const void *Src, size_t Count, event DepEvent,
                    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->memcpy(impl, Dest, Src, Count, {DepEvent},
                      /*CallerNeedsEvent=*/true, TlsCodeLocCapture.query());
}

event queue::memcpy(void *Dest, const void *Src, size_t Count,
                    const std::vector<event> &DepEvents,
                    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->memcpy(impl, Dest, Src, Count, DepEvents,
                      /*CallerNeedsEvent=*/true, TlsCodeLocCapture.query());
}

event queue::mem_advise(const void *Ptr, size_t Length, int Advice,
                        const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->mem_advise(impl, Ptr, Length, ur_usm_advice_flags_t(Advice), {},
                          /*CallerNeedsEvent=*/true);
}

event queue::mem_advise(const void *Ptr, size_t Length, int Advice,
                        event DepEvent, const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->mem_advise(impl, Ptr, Length, ur_usm_advice_flags_t(Advice),
                          {DepEvent},
                          /*CallerNeedsEvent=*/true);
}

event queue::mem_advise(const void *Ptr, size_t Length, int Advice,
                        const std::vector<event> &DepEvents,
                        const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->mem_advise(impl, Ptr, Length, ur_usm_advice_flags_t(Advice),
                          DepEvents,
                          /*CallerNeedsEvent=*/true);
}

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
/// TODO: Unused. Remove these when ABI-break window is open.
event queue::submit_impl(std::function<void(handler &)> CGH,
                         const detail::code_location &CodeLoc) {
  return submit_with_event_impl(CGH, {}, CodeLoc, true);
}
event queue::submit_impl(std::function<void(handler &)> CGH,
                         const detail::code_location &CodeLoc,
                         bool IsTopCodeLoc) {
  return submit_with_event_impl(CGH, {}, CodeLoc, IsTopCodeLoc);
}

event queue::submit_impl(std::function<void(handler &)> CGH, queue SecondQueue,
                         const detail::code_location &CodeLoc) {
  return impl->submit(CGH, impl, SecondQueue.impl, CodeLoc, true);
}
event queue::submit_impl(std::function<void(handler &)> CGH, queue SecondQueue,
                         const detail::code_location &CodeLoc,
                         bool IsTopCodeLoc) {
  return impl->submit(CGH, impl, SecondQueue.impl, CodeLoc, IsTopCodeLoc);
}

void queue::submit_without_event_impl(std::function<void(handler &)> CGH,
                                      const detail::code_location &CodeLoc) {
  submit_without_event_impl(CGH, {}, CodeLoc, true);
}
void queue::submit_without_event_impl(std::function<void(handler &)> CGH,
                                      const detail::code_location &CodeLoc,
                                      bool IsTopCodeLoc) {
  submit_without_event_impl(CGH, {}, CodeLoc, IsTopCodeLoc);
}

event queue::submit_impl_and_postprocess(
    std::function<void(handler &)> CGH, const detail::code_location &CodeLoc,
    const detail::SubmitPostProcessF &PostProcess) {
  detail::SubmissionInfo SI{};
  SI.PostProcessorFunc() = std::move(PostProcess);
  return submit_with_event_impl(CGH, SI, CodeLoc, true);
}
event queue::submit_impl_and_postprocess(
    std::function<void(handler &)> CGH, const detail::code_location &CodeLoc,
    const detail::SubmitPostProcessF &PostProcess, bool IsTopCodeLoc) {
  detail::SubmissionInfo SI{};
  SI.PostProcessorFunc() = std::move(PostProcess);
  return submit_with_event_impl(CGH, SI, CodeLoc, IsTopCodeLoc);
}

event queue::submit_impl_and_postprocess(
    std::function<void(handler &)> CGH, queue SecondQueue,
    const detail::code_location &CodeLoc,
    const detail::SubmitPostProcessF &PostProcess) {
  return impl->submit(CGH, impl, SecondQueue.impl, CodeLoc, true, &PostProcess);
}
event queue::submit_impl_and_postprocess(
    std::function<void(handler &)> CGH, queue SecondQueue,
    const detail::code_location &CodeLoc,
    const detail::SubmitPostProcessF &PostProcess, bool IsTopCodeLoc) {
  return impl->submit(CGH, impl, SecondQueue.impl, CodeLoc, IsTopCodeLoc,
                      &PostProcess);
}
#endif // __INTEL_PREVIEW_BREAKING_CHANGES

event queue::submit_with_event_impl(std::function<void(handler &)> CGH,
                                    const detail::SubmissionInfo &SubmitInfo,
                                    const detail::code_location &CodeLoc,
                                    bool IsTopCodeLoc) {
  return impl->submit_with_event(CGH, impl, SubmitInfo, CodeLoc, IsTopCodeLoc);
}

void queue::submit_without_event_impl(std::function<void(handler &)> CGH,
                                      const detail::SubmissionInfo &SubmitInfo,
                                      const detail::code_location &CodeLoc,
                                      bool IsTopCodeLoc) {
  impl->submit_without_event(CGH, impl, SubmitInfo, CodeLoc, IsTopCodeLoc);
}

void queue::wait_proxy(const detail::code_location &CodeLoc) {
  impl->wait(CodeLoc);
}

void queue::wait_and_throw_proxy(const detail::code_location &CodeLoc) {
  impl->wait_and_throw(CodeLoc);
}

static event
getBarrierEventForInorderQueueHelper(const detail::QueueImplPtr QueueImpl) {
  // This function should not be called when a queue is recording to a graph,
  // as a graph can record from multiple queues and we cannot guarantee the
  // last node added by an in-order queue will be the last node added to the
  // graph.
  assert(!QueueImpl->getCommandGraph() &&
         "Should not be called in on graph recording.");

  return QueueImpl->getLastEvent();
}

/// Prevents any commands submitted afterward to this queue from executing
/// until all commands previously submitted to this queue have entered the
/// complete state.
///
/// \param CodeLoc is the code location of the submit call (default argument)
/// \return a SYCL event object, which corresponds to the queue the command
/// group is being enqueued on.
event queue::ext_oneapi_submit_barrier(const detail::code_location &CodeLoc) {
  if (is_in_order() && !impl->getCommandGraph() && !impl->MDiscardEvents &&
      !impl->MIsProfilingEnabled) {
    event InOrderLastEvent = getBarrierEventForInorderQueueHelper(impl);
    // If the last event was discarded, fall back to enqueuing a barrier.
    if (!detail::getSyclObjImpl(InOrderLastEvent)->isDiscarded())
      return InOrderLastEvent;
  }

  return submit([=](handler &CGH) { CGH.ext_oneapi_barrier(); }, CodeLoc);
}

/// Prevents any commands submitted afterward to this queue from executing
/// until all events in WaitList have entered the complete state. If WaitList
/// is empty, then ext_oneapi_submit_barrier has no effect.
///
/// \param WaitList is a vector of valid SYCL events that need to complete
/// before barrier command can be executed.
/// \param CodeLoc is the code location of the submit call (default argument)
/// \return a SYCL event object, which corresponds to the queue the command
/// group is being enqueued on.
event queue::ext_oneapi_submit_barrier(const std::vector<event> &WaitList,
                                       const detail::code_location &CodeLoc) {
  bool AllEventsEmptyOrNop = std::all_of(
      begin(WaitList), end(WaitList), [&](const event &Event) -> bool {
        auto EventImpl = detail::getSyclObjImpl(Event);
        return (EventImpl->isDefaultConstructed() || EventImpl->isNOP()) &&
               !EventImpl->getCommandGraph();
      });
  if (is_in_order() && !impl->getCommandGraph() && !impl->MDiscardEvents &&
      !impl->MIsProfilingEnabled && AllEventsEmptyOrNop) {
    event InOrderLastEvent = getBarrierEventForInorderQueueHelper(impl);
    // If the last event was discarded, fall back to enqueuing a barrier.
    if (!detail::getSyclObjImpl(InOrderLastEvent)->isDiscarded())
      return InOrderLastEvent;
  }

  return submit([=](handler &CGH) { CGH.ext_oneapi_barrier(WaitList); },
                CodeLoc);
}

template <typename Param>
typename detail::is_queue_info_desc<Param>::return_type
queue::get_info() const {
  return impl->get_info<Param>();
}

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, Picode)              \
  template __SYCL_EXPORT ReturnT queue::get_info<info::queue::Desc>() const;

#include <sycl/info/queue_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

template <typename Param>
typename detail::is_backend_info_desc<Param>::return_type
queue::get_backend_info() const {
  return impl->get_backend_info<Param>();
}

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, Picode)              \
  template __SYCL_EXPORT ReturnT                                               \
  queue::get_backend_info<info::DescType::Desc>() const;

#include <sycl/info/sycl_backend_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

bool queue::is_in_order() const {
  return has_property<property::queue::in_order>();
}

backend queue::get_backend() const noexcept { return getImplBackend(impl); }

bool queue::ext_oneapi_empty() const { return impl->ext_oneapi_empty(); }

void queue::ext_oneapi_prod() { impl->flush(); }

ur_native_handle_t queue::getNative(int32_t &NativeHandleDesc) const {
  return impl->getNative(NativeHandleDesc);
}

event queue::memcpyToDeviceGlobal(void *DeviceGlobalPtr, const void *Src,
                                  bool IsDeviceImageScope, size_t NumBytes,
                                  size_t Offset,
                                  const std::vector<event> &DepEvents) {
  return impl->memcpyToDeviceGlobal(impl, DeviceGlobalPtr, Src,
                                    IsDeviceImageScope, NumBytes, Offset,
                                    DepEvents, /*CallerNeedsEvent=*/true);
}

event queue::memcpyFromDeviceGlobal(void *Dest, const void *DeviceGlobalPtr,
                                    bool IsDeviceImageScope, size_t NumBytes,
                                    size_t Offset,
                                    const std::vector<event> &DepEvents) {
  return impl->memcpyFromDeviceGlobal(impl, Dest, DeviceGlobalPtr,
                                      IsDeviceImageScope, NumBytes, Offset,
                                      DepEvents, /*CallerNeedsEvent=*/true);
}

bool queue::device_has(aspect Aspect) const {
  // avoid creating sycl object from impl
  return impl->getDeviceImplPtr()->has(Aspect);
}

// TODO(#15184) Remove this function in the next ABI-breaking window.
bool queue::ext_codeplay_supports_fusion() const { return false; }

event queue::ext_oneapi_get_last_event() const {
  if (!is_in_order())
    throw sycl::exception(
        make_error_code(errc::invalid),
        "ext_oneapi_get_last_event() can only be called on in-order queues.");

  event LastEvent = impl->getLastEvent();
  // If the last event was discarded or a NOP, we insert a marker to represent
  // an event at end.
  auto LastEventImpl = detail::getSyclObjImpl(LastEvent);
  if (LastEventImpl->isDiscarded() || LastEventImpl->isNOP())
    LastEvent =
        detail::createSyclObjFromImpl<event>(impl->insertMarkerEvent(impl));
  return LastEvent;
}

void queue::ext_oneapi_set_external_event(const event &external_event) {
  if (!is_in_order())
    throw sycl::exception(make_error_code(errc::invalid),
                          "ext_oneapi_set_external_event() can only be called "
                          "on in-order queues.");
  return impl->setExternalEvent(external_event);
}

const property_list &queue::getPropList() const { return impl->getPropList(); }

} // namespace _V1
} // namespace sycl

size_t std::hash<sycl::queue>::operator()(const sycl::queue &Q) const {
  // Compared to using the impl pointer, the unique ID helps avoid hash
  // collisions with previously destroyed queues.
  return std::hash<unsigned long long>()(
      sycl::detail::getSyclObjImpl(Q)->getQueueID());
}
