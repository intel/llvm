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
#include <sycl/ext/codeplay/experimental/fusion_properties.hpp>
#include <sycl/handler.hpp>
#include <sycl/queue.hpp>
#include <sycl/stl.hpp>

#include <algorithm>

namespace sycl {
inline namespace _V1 {

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
      reinterpret_cast<sycl::detail::pi::PiQueue>(clQueue),
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

bool queue::is_host() const {
  bool IsHost = impl->is_host();
  assert(!IsHost && "queue::is_host should not be called in implementation.");
  return IsHost;
}

void queue::throw_asynchronous() { impl->throw_asynchronous(); }

event queue::memset(void *Ptr, int Value, size_t Count,
                    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->memset(impl, Ptr, Value, Count, {});
}

event queue::memset(void *Ptr, int Value, size_t Count, event DepEvent,
                    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->memset(impl, Ptr, Value, Count, {DepEvent});
}

event queue::memset(void *Ptr, int Value, size_t Count,
                    const std::vector<event> &DepEvents,
                    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->memset(impl, Ptr, Value, Count, DepEvents);
}

event queue::memcpy(void *Dest, const void *Src, size_t Count,
                    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->memcpy(impl, Dest, Src, Count, {}, CodeLoc);
}

event queue::memcpy(void *Dest, const void *Src, size_t Count, event DepEvent,
                    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->memcpy(impl, Dest, Src, Count, {DepEvent}, CodeLoc);
}

event queue::memcpy(void *Dest, const void *Src, size_t Count,
                    const std::vector<event> &DepEvents,
                    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->memcpy(impl, Dest, Src, Count, DepEvents, CodeLoc);
}

event queue::mem_advise(const void *Ptr, size_t Length, pi_mem_advice Advice,
                        const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return mem_advise(Ptr, Length, int(Advice));
}

event queue::mem_advise(const void *Ptr, size_t Length, int Advice,
                        const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->mem_advise(impl, Ptr, Length, pi_mem_advice(Advice), {});
}

event queue::mem_advise(const void *Ptr, size_t Length, int Advice,
                        event DepEvent, const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->mem_advise(impl, Ptr, Length, pi_mem_advice(Advice), {DepEvent});
}

event queue::mem_advise(const void *Ptr, size_t Length, int Advice,
                        const std::vector<event> &DepEvents,
                        const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return impl->mem_advise(impl, Ptr, Length, pi_mem_advice(Advice), DepEvents);
}

event queue::discard_or_return(const event &Event) {
  if (!(impl->MDiscardEvents))
    return Event;
  using detail::event_impl;
  auto Impl = std::make_shared<event_impl>(event_impl::HES_Discarded);
  return detail::createSyclObjFromImpl<event>(Impl);
}

event queue::submit_impl(std::function<void(handler &)> CGH,
                         const detail::code_location &CodeLoc) {
  return impl->submit(CGH, impl, CodeLoc);
}

event queue::submit_impl(std::function<void(handler &)> CGH, queue SecondQueue,
                         const detail::code_location &CodeLoc) {
  return impl->submit(CGH, impl, SecondQueue.impl, CodeLoc);
}

event queue::submit_impl_and_postprocess(
    std::function<void(handler &)> CGH, const detail::code_location &CodeLoc,
    const SubmitPostProcessF &PostProcess) {
  return impl->submit(CGH, impl, CodeLoc, &PostProcess);
}

event queue::submit_impl_and_postprocess(
    std::function<void(handler &)> CGH, queue SecondQueue,
    const detail::code_location &CodeLoc,
    const SubmitPostProcessF &PostProcess) {
  return impl->submit(CGH, impl, SecondQueue.impl, CodeLoc, &PostProcess);
}

void queue::wait_proxy(const detail::code_location &CodeLoc) {
  impl->wait(CodeLoc);
}

void queue::wait_and_throw_proxy(const detail::code_location &CodeLoc) {
  impl->wait_and_throw(CodeLoc);
}

/// Prevents any commands submitted afterward to this queue from executing
/// until all commands previously submitted to this queue have entered the
/// complete state.
///
/// \param CodeLoc is the code location of the submit call (default argument)
/// \return a SYCL event object, which corresponds to the queue the command
/// group is being enqueued on.
event queue::ext_oneapi_submit_barrier(const detail::code_location &CodeLoc) {
  if (is_in_order()) {
    // The last command recorded in the graph is not tracked by the queue but by
    // the graph itself. We must therefore search for the last node/event in the
    // graph.
    if (auto Graph = impl->getCommandGraph()) {
      auto LastEvent = Graph->getEventForNode(Graph->getLastInorderNode(impl));
      return sycl::detail::createSyclObjFromImpl<event>(LastEvent);
    }
    return impl->getLastEvent();
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
  if (is_in_order() && WaitList.empty()) {
    // The last command recorded in the graph is not tracked by the queue but by
    // the graph itself. We must therefore search for the last node/event in the
    // graph.
    if (auto Graph = impl->getCommandGraph()) {
      auto LastEvent = Graph->getEventForNode(Graph->getLastInorderNode(impl));
      return sycl::detail::createSyclObjFromImpl<event>(LastEvent);
    }
    return impl->getLastEvent();
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

template <typename PropertyT> bool queue::has_property() const noexcept {
  return impl->has_property<PropertyT>();
}

template <typename PropertyT> PropertyT queue::get_property() const {
  return impl->get_property<PropertyT>();
}

#define __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)                  \
  template __SYCL_EXPORT bool queue::has_property<NS_QUALIFIER::PROP_NAME>()   \
      const noexcept;                                                          \
  template __SYCL_EXPORT NS_QUALIFIER::PROP_NAME                               \
  queue::get_property<NS_QUALIFIER::PROP_NAME>() const;

#define __SYCL_DATA_LESS_PROP(NS_QUALIFIER, PROP_NAME, ENUM_VAL)               \
  __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)

#include <sycl/properties/queue_properties.def>

bool queue::is_in_order() const {
  return impl->has_property<property::queue::in_order>();
}

backend queue::get_backend() const noexcept { return getImplBackend(impl); }

bool queue::ext_oneapi_empty() const { return impl->ext_oneapi_empty(); }

pi_native_handle queue::getNative(int32_t &NativeHandleDesc) const {
  return impl->getNative(NativeHandleDesc);
}

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
buffer<detail::AssertHappened, 1> &queue::getAssertHappenedBuffer() {
  return impl->getAssertHappenedBuffer();
}
#endif

event queue::memcpyToDeviceGlobal(void *DeviceGlobalPtr, const void *Src,
                                  bool IsDeviceImageScope, size_t NumBytes,
                                  size_t Offset,
                                  const std::vector<event> &DepEvents) {
  return impl->memcpyToDeviceGlobal(impl, DeviceGlobalPtr, Src,
                                    IsDeviceImageScope, NumBytes, Offset,
                                    DepEvents);
}

event queue::memcpyFromDeviceGlobal(void *Dest, const void *DeviceGlobalPtr,
                                    bool IsDeviceImageScope, size_t NumBytes,
                                    size_t Offset,
                                    const std::vector<event> &DepEvents) {
  return impl->memcpyFromDeviceGlobal(impl, Dest, DeviceGlobalPtr,
                                      IsDeviceImageScope, NumBytes, Offset,
                                      DepEvents);
}

bool queue::device_has(aspect Aspect) const {
  // avoid creating sycl object from impl
  return impl->getDeviceImplPtr()->has(Aspect);
}

bool queue::ext_codeplay_supports_fusion() const {
  return impl->has_property<
      ext::codeplay::experimental::property::queue::enable_fusion>();
}

event queue::ext_oneapi_get_last_event() const {
  if (!is_in_order())
    throw sycl::exception(
        make_error_code(errc::invalid),
        "ext_oneapi_get_last_event() can only be called on in-order queues.");
  if (impl->MDiscardEvents)
    throw sycl::exception(
        make_error_code(errc::invalid),
        "ext_oneapi_get_last_event() cannot be called on queues with the "
        "ext::oneapi::property::queue::discard_events property.");
  return impl->getLastEvent();
}

void queue::ext_oneapi_set_external_event(const event &external_event) {
  if (!is_in_order())
    throw sycl::exception(make_error_code(errc::invalid),
                          "ext_oneapi_set_external_event() can only be called "
                          "on in-order queues.");
  if (impl->MDiscardEvents)
    throw sycl::exception(
        make_error_code(errc::invalid),
        "ext_oneapi_set_external_event() cannot be called on queues with the "
        "ext::oneapi::property::queue::discard_events property.");
  return impl->setExternalEvent(external_event);
}

} // namespace _V1
} // namespace sycl
