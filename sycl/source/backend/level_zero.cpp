//==--------- level_zero.cpp - SYCL Level-Zero backend ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <detail/platform_impl.hpp>
#include <detail/plugin.hpp>
#include <detail/program_impl.hpp>
#include <detail/queue_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace level_zero {
using namespace detail;

//----------------------------------------------------------------------------
// Implementation of level_zero::make<platform>
__SYCL_EXPORT platform make_platform(pi_native_handle NativeHandle) {
  const auto &Plugin = pi::getPlugin<backend::level_zero>();
  // Create PI platform first.
  pi::PiPlatform PiPlatform;
  Plugin.call<PiApiKind::piextPlatformCreateWithNativeHandle>(NativeHandle,
                                                              &PiPlatform);

  return detail::createSyclObjFromImpl<platform>(
      platform_impl::getOrMakePlatformImpl(PiPlatform, Plugin));
}

//----------------------------------------------------------------------------
// Implementation of level_zero::make<device>
__SYCL_EXPORT device make_device(const platform &Platform,
                                 pi_native_handle NativeHandle) {
  const auto &Plugin = pi::getPlugin<backend::level_zero>();
  const auto &PlatformImpl = getSyclObjImpl(Platform);
  // Create PI device first.
  pi::PiDevice PiDevice;
  Plugin.call<PiApiKind::piextDeviceCreateWithNativeHandle>(
      NativeHandle, PlatformImpl->getHandleRef(), &PiDevice);

  return detail::createSyclObjFromImpl<device>(
      PlatformImpl->getOrMakeDeviceImpl(PiDevice, PlatformImpl));
}

//----------------------------------------------------------------------------
// Implementation of level_zero::make<context>
__SYCL_EXPORT context make_context(const vector_class<device> &DeviceList,
                                   pi_native_handle NativeHandle) {
  const auto &Plugin = pi::getPlugin<backend::level_zero>();
  // Create PI context first.
  pi_context PiContext;
  vector_class<pi_device> DeviceHandles;
  for (auto Dev : DeviceList) {
    DeviceHandles.push_back(detail::getSyclObjImpl(Dev)->getHandleRef());
  }
  Plugin.call<PiApiKind::piextContextCreateWithNativeHandle>(
      NativeHandle, DeviceHandles.size(), DeviceHandles.data(), &PiContext);
  // Construct the SYCL context from PI context.
  return detail::createSyclObjFromImpl<context>(
      std::make_shared<context_impl>(PiContext, async_handler{}, Plugin));
}

//----------------------------------------------------------------------------
// Implementation of level_zero::make<program>
__SYCL_EXPORT program make_program(const context &Context,
                                   pi_native_handle NativeHandle) {
  // Construct the SYCL program from native program.
  // TODO: move here the code that creates PI program, and remove the
  // native interop constructor.
  return detail::createSyclObjFromImpl<program>(
      std::make_shared<program_impl>(getSyclObjImpl(Context), NativeHandle));
}

//----------------------------------------------------------------------------
// Implementation of level_zero::make<queue>
__SYCL_EXPORT queue make_queue(const context &Context,
                               pi_native_handle NativeHandle) {
  const auto &Plugin = pi::getPlugin<backend::level_zero>();
  const auto &ContextImpl = getSyclObjImpl(Context);
  // Create PI queue first.
  pi::PiQueue PiQueue;
  Plugin.call<PiApiKind::piextQueueCreateWithNativeHandle>(
      NativeHandle, ContextImpl->getHandleRef(), &PiQueue);
  // Construct the SYCL queue from PI queue.
  return detail::createSyclObjFromImpl<queue>(std::make_shared<queue_impl>(
      PiQueue, ContextImpl, ContextImpl->get_async_handler()));
}

} // namespace level_zero
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
