//==------------------- backend.cpp ----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "detail/context_impl.hpp"
#include "detail/event_impl.hpp"
#include "detail/platform_impl.hpp"
#include "detail/plugin.hpp"
#include "detail/queue_impl.hpp"
#include <CL/sycl/backend.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/pi.h>
#include <CL/sycl/exception_list.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

static const plugin &getPlugin(backend Backend) {
  switch (Backend) {
  case backend::opencl:
    return pi::getPlugin<backend::opencl>();
  case backend::level_zero:
    return pi::getPlugin<backend::level_zero>();
  default:
    throw sycl::runtime_error{"Unsupported backend", PI_INVALID_OPERATION};
  }
}

platform make_platform(pi_native_handle NativeHandle, backend Backend) {
  const auto &Plugin = getPlugin(Backend);

  // Create PI platform first.
  pi::PiPlatform PiPlatform = nullptr;
  Plugin.call<PiApiKind::piextPlatformCreateWithNativeHandle>(NativeHandle,
                                                              &PiPlatform);

  return detail::createSyclObjFromImpl<platform>(
      platform_impl::getOrMakePlatformImpl(PiPlatform, Plugin));
}

__SYCL_EXPORT device make_device(pi_native_handle NativeHandle,
                                 backend Backend) {
  const auto &Plugin = getPlugin(Backend);

  pi::PiDevice PiDevice = nullptr;
  Plugin.call<PiApiKind::piextDeviceCreateWithNativeHandle>(NativeHandle,
                                                            nullptr, &PiDevice);
  // Construct the SYCL device from PI device.
  return detail::createSyclObjFromImpl<device>(
      std::make_shared<device_impl>(PiDevice, Plugin));
}

__SYCL_EXPORT context make_context(pi_native_handle NativeHandle,
                                   const async_handler &Handler,
                                   backend Backend) {
  const auto &Plugin = getPlugin(Backend);

  pi::PiContext PiContext = nullptr;
  Plugin.call<PiApiKind::piextContextCreateWithNativeHandle>(
      NativeHandle, 0, nullptr, false, &PiContext);
  // Construct the SYCL context from PI context.
  return detail::createSyclObjFromImpl<context>(
      std::make_shared<context_impl>(PiContext, Handler, Plugin));
}

__SYCL_EXPORT queue make_queue(pi_native_handle NativeHandle,
                               const context &Context,
                               const async_handler &Handler, backend Backend) {
  const auto &Plugin = getPlugin(Backend);
  const auto &ContextImpl = getSyclObjImpl(Context);
  // Create PI queue first.
  pi::PiQueue PiQueue = nullptr;
  Plugin.call<PiApiKind::piextQueueCreateWithNativeHandle>(
      NativeHandle, ContextImpl->getHandleRef(), &PiQueue);
  // Construct the SYCL queue from PI queue.
  return detail::createSyclObjFromImpl<queue>(
      std::make_shared<queue_impl>(PiQueue, ContextImpl, Handler));
}

__SYCL_EXPORT event make_event(pi_native_handle NativeHandle,
                               const context &Context, backend Backend) {
  const auto &Plugin = getPlugin(Backend);

  pi::PiEvent PiEvent = nullptr;
  Plugin.call<PiApiKind::piextEventCreateWithNativeHandle>(NativeHandle,
                                                           &PiEvent);

  return detail::createSyclObjFromImpl<event>(
      std::make_shared<event_impl>(PiEvent, Context));
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
