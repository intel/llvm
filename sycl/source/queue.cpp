//==-------------- queue.cpp -----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/event.hpp>
#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/queue.hpp>
#include <CL/sycl/stl.hpp>
#include <detail/backend_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/queue_impl.hpp>

#include <algorithm>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

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

queue::queue(cl_command_queue clQueue, const context &SyclContext,
             const async_handler &AsyncHandler) {
  impl = std::make_shared<detail::queue_impl>(
      reinterpret_cast<RT::PiQueue>(clQueue),
      detail::getSyclObjImpl(SyclContext), AsyncHandler);
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

cl_command_queue queue::get() const { return impl->get(); }

context queue::get_context() const { return impl->get_context(); }

device queue::get_device() const { return impl->get_device(); }

bool queue::is_host() const { return impl->is_host(); }


void queue::throw_asynchronous() { impl->throw_asynchronous(); }

event queue::memset(void *Ptr, int Value, size_t Count) {
  return impl->memset(impl, Ptr, Value, Count, {});
}

event queue::memset(void *Ptr, int Value, size_t Count, event DepEvent) {
  return impl->memset(impl, Ptr, Value, Count, {DepEvent});
}

event queue::memset(void *Ptr, int Value, size_t Count,
                    const std::vector<event> &DepEvents) {
  return impl->memset(impl, Ptr, Value, Count, DepEvents);
}

event queue::memcpy(void *Dest, const void *Src, size_t Count) {
  return impl->memcpy(impl, Dest, Src, Count, {});
}

event queue::memcpy(void *Dest, const void *Src, size_t Count, event DepEvent) {
  return impl->memcpy(impl, Dest, Src, Count, {DepEvent});
}

event queue::memcpy(void *Dest, const void *Src, size_t Count,
                    const std::vector<event> &DepEvents) {
  return impl->memcpy(impl, Dest, Src, Count, DepEvents);
}

event queue::mem_advise(const void *Ptr, size_t Length, pi_mem_advice Advice) {
  return mem_advise(Ptr, Length, int(Advice));
}

event queue::mem_advise(const void *Ptr, size_t Length, int Advice) {
  return impl->mem_advise(impl, Ptr, Length, pi_mem_advice(Advice), {});
}

event queue::mem_advise(const void *Ptr, size_t Length, int Advice,
                        event DepEvent) {
  return impl->mem_advise(impl, Ptr, Length, pi_mem_advice(Advice), {DepEvent});
}

event queue::mem_advise(const void *Ptr, size_t Length, int Advice,
                        const std::vector<event> &DepEvents) {
  return impl->mem_advise(impl, Ptr, Length, pi_mem_advice(Advice), DepEvents);
}

event queue::discard_or_return(const event &Event) {
  if (impl->MDiscardEvents) {
    using detail::event_impl;
    auto Impl = std::make_shared<event_impl>(event_impl::HES_Discarded);
    return detail::createSyclObjFromImpl<event>(Impl);
  }
  return Event;
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

template <info::queue Param>
typename info::param_traits<info::queue, Param>::return_type
queue::get_info() const {
  return impl->get_info<Param>();
}

#define __SYCL_PARAM_TRAITS_SPEC(ParamType, Param, RetType)                    \
  template __SYCL_EXPORT RetType queue::get_info<info::ParamType::Param>()     \
      const;

#include <CL/sycl/info/queue_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

template <typename PropertyT> bool queue::has_property() const {
  return impl->has_property<PropertyT>();
}

template <typename PropertyT> PropertyT queue::get_property() const {
  return impl->get_property<PropertyT>();
}

template __SYCL_EXPORT bool
queue::has_property<property::queue::enable_profiling>() const;
template __SYCL_EXPORT property::queue::enable_profiling
queue::get_property<property::queue::enable_profiling>() const;

bool queue::is_in_order() const {
  return impl->has_property<property::queue::in_order>();
}

backend queue::get_backend() const noexcept { return getImplBackend(impl); }

pi_native_handle queue::getNative() const { return impl->getNative(); }

buffer<detail::AssertHappened, 1> &queue::getAssertHappenedBuffer() {
  return impl->getAssertHappenedBuffer();
}

bool queue::device_has(aspect Aspect) const {
  // avoid creating sycl object from impl
  return impl->getDeviceImplPtr()->has(Aspect);
}
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
