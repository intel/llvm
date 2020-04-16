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
#include <detail/queue_impl.hpp>

#include <algorithm>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

namespace detail {

QueueOrder getQueueOrder(const property_list &propList) {
  if (propList.has_property<property::queue::in_order>()) {
    return QueueOrder::Ordered;
  }
  return QueueOrder::OOO;
}

} // namespace detail

queue::queue(const context &syclContext, const device_selector &deviceSelector,
             const async_handler &asyncHandler, const property_list &propList) {

  const vector_class<device> Devs = syclContext.get_devices();

  auto Comp = [&deviceSelector](const device &d1, const device &d2) {
    return deviceSelector(d1) < deviceSelector(d2);
  };

  const device &syclDevice = *std::max_element(Devs.begin(), Devs.end(), Comp);

  impl = std::make_shared<detail::queue_impl>(
      detail::getSyclObjImpl(syclDevice), detail::getSyclObjImpl(syclContext),
      asyncHandler, detail::getQueueOrder(propList), propList);
}

queue::queue(const context &syclContext,
             const device &syclDevice,
             const async_handler &asyncHandler,
             const property_list &propList) {
  impl = std::make_shared<detail::queue_impl>(
      detail::getSyclObjImpl(syclDevice), detail::getSyclObjImpl(syclContext),
      asyncHandler, cl::sycl::detail::QueueOrder::OOO, propList);
}

queue::queue(const device &syclDevice, const async_handler &asyncHandler,
             const property_list &propList) {
  impl = std::make_shared<detail::queue_impl>(
      detail::getSyclObjImpl(syclDevice), asyncHandler,
      detail::getQueueOrder(propList), propList);
}

queue::queue(cl_command_queue clQueue, const context &syclContext,
             const async_handler &asyncHandler) {
  impl = std::make_shared<detail::queue_impl>(
      detail::pi::cast<detail::RT::PiQueue>(clQueue),
      detail::getSyclObjImpl(syclContext), asyncHandler);
}

queue::queue(const context &syclContext, const device_selector &deviceSelector,
             const property_list &propList)
    : queue(syclContext, deviceSelector,
            detail::getSyclObjImpl(syclContext)->get_async_handler(),
            propList) {}

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

event queue::memset(void *ptr, int value, size_t count) {
  return impl->memset(impl, ptr, value, count);
}

event queue::memcpy(void *dest, const void *src, size_t count) {
  return impl->memcpy(impl, dest, src, count);
}

event queue::mem_advise(const void *ptr, size_t length, pi_mem_advice advice) {
  return impl->mem_advise(ptr, length, advice);
}

event queue::submit_impl(function_class<void(handler &)> CGH,
                         const detail::code_location &CodeLoc) {
  return impl->submit(CGH, impl, CodeLoc);
}

event queue::submit_impl(function_class<void(handler &)> CGH, queue secondQueue,
                         const detail::code_location &CodeLoc) {
  return impl->submit(CGH, impl, secondQueue.impl, CodeLoc);
}

void queue::wait_proxy(const detail::code_location &CodeLoc) {
  impl->wait(CodeLoc);
}

void queue::wait_and_throw_proxy(const detail::code_location &CodeLoc) {
  impl->wait_and_throw(CodeLoc);
}

template <info::queue param>
typename info::param_traits<info::queue, param>::return_type
queue::get_info() const {
  return impl->get_info<param>();
}

#define PARAM_TRAITS_SPEC(param_type, param, ret_type)                         \
  template __SYCL_EXPORT ret_type queue::get_info<info::param_type::param>()   \
      const;

#include <CL/sycl/info/queue_traits.def>

#undef PARAM_TRAITS_SPEC

template <typename propertyT> bool queue::has_property() const {
  return impl->has_property<propertyT>();
}

template <typename propertyT> propertyT queue::get_property() const {
  return impl->get_property<propertyT>();
}

template __SYCL_EXPORT bool
queue::has_property<property::queue::enable_profiling>() const;
template __SYCL_EXPORT property::queue::enable_profiling
queue::get_property<property::queue::enable_profiling>() const;

bool queue::is_in_order() const {
  return impl->has_property<property::queue::in_order>();
}

pi_native_handle queue::getNative() const { return impl->getNative(); }

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
