//==-------------- ordered_queue.cpp ---------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/device.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/ordered_queue.hpp>
#include <detail/queue_impl.hpp>

#include <algorithm>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
ordered_queue::ordered_queue(const context &syclContext,
                             const device_selector &deviceSelector,
                             const property_list &propList)
    : ordered_queue(syclContext, deviceSelector,
                    detail::getSyclObjImpl(syclContext)->get_async_handler(),
                    propList) {}
ordered_queue::ordered_queue(const context &syclContext,
                             const device_selector &deviceSelector,
                             const async_handler &asyncHandler,
                             const property_list &propList) {

  const vector_class<device> Devs = syclContext.get_devices();

  auto Comp = [&deviceSelector](const device &d1, const device &d2) {
    return deviceSelector(d1) < deviceSelector(d2);
  };

  const device &syclDevice = *std::max_element(Devs.begin(), Devs.end(), Comp);
  impl = std::make_shared<detail::queue_impl>(
      detail::getSyclObjImpl(syclDevice), detail::getSyclObjImpl(syclContext),
      asyncHandler, cl::sycl::detail::QueueOrder::Ordered, propList);
}

ordered_queue::ordered_queue(const device &syclDevice,
                             const async_handler &asyncHandler,
                             const property_list &propList) {
  impl = std::make_shared<detail::queue_impl>(
      detail::getSyclObjImpl(syclDevice), asyncHandler,
      cl::sycl::detail::QueueOrder::Ordered, propList);
}

ordered_queue::ordered_queue(cl_command_queue clQueue,
                             const context &syclContext,
                             const async_handler &asyncHandler) {
  cl_command_queue_properties reportedProps;
  RT::PiQueue m_CommandQueue = detail::pi::cast<detail::RT::PiQueue>(clQueue);
  const detail::plugin &Plugin = detail::getSyclObjImpl(syclContext)->getPlugin();
  Plugin.call<detail::PiApiKind::piQueueGetInfo>(
      m_CommandQueue, PI_QUEUE_INFO_DEVICE, sizeof(reportedProps),
      &reportedProps, nullptr);
  if (reportedProps & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
    throw runtime_error(
        "Failed to build a sycl ordered queue from a cl OOO queue.",
        PI_INVALID_OPERATION);

  impl = std::make_shared<detail::queue_impl>(
      m_CommandQueue, detail::getSyclObjImpl(syclContext), asyncHandler);
}

cl_command_queue ordered_queue::get() const { return impl->get(); }

context ordered_queue::get_context() const { return impl->get_context(); }

device ordered_queue::get_device() const { return impl->get_device(); }

bool ordered_queue::is_host() const { return impl->is_host(); }

void ordered_queue::throw_asynchronous() { impl->throw_asynchronous(); }

event ordered_queue::memset(void *ptr, int value, size_t count) {
  return impl->memset(impl, ptr, value, count);
}

event ordered_queue::memcpy(void *dest, const void *src, size_t count) {
  return impl->memcpy(impl, dest, src, count);
}

event ordered_queue::submit_impl(function_class<void(handler &)> CGH,
                                 const detail::code_location &CodeLoc) {
  return impl->submit(CGH, impl, CodeLoc);
}

event ordered_queue::submit_impl(function_class<void(handler &)> CGH,
                                 ordered_queue &secondQueue,
                                 const detail::code_location &CodeLoc) {
  return impl->submit(CGH, impl, secondQueue.impl, CodeLoc);
}

void ordered_queue::wait_proxy(const detail::code_location &CodeLoc) {
  impl->wait(CodeLoc);
}

void ordered_queue::wait_and_throw_proxy(const detail::code_location &CodeLoc) {
  impl->wait_and_throw(CodeLoc);
}

template <info::queue param>
typename info::param_traits<info::queue, param>::return_type
ordered_queue::get_info() const {
  return impl->get_info<param>();
}

#define PARAM_TRAITS_SPEC(param_type, param, ret_type)                         \
  template __SYCL_EXPORT ret_type                                              \
  ordered_queue::get_info<info::param_type::param>() const;

#include <CL/sycl/info/queue_traits.def>

#undef PARAM_TRAITS_SPEC

template <typename propertyT> bool ordered_queue::has_property() const {
  return impl->has_property<propertyT>();
}

template <typename propertyT> propertyT ordered_queue::get_property() const {
  return impl->get_property<propertyT>();
}

template __SYCL_EXPORT bool
ordered_queue::has_property<property::queue::enable_profiling>() const;
template __SYCL_EXPORT property::queue::enable_profiling
ordered_queue::get_property<property::queue::enable_profiling>() const;
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
