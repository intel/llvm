//==--------- level_zero.cpp - SYCL Level-Zero backend ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <CL/sycl/backend.hpp>
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
  return detail::make_platform(NativeHandle, backend::level_zero);
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
__SYCL_EXPORT context make_context(const std::vector<device> &DeviceList,
                                   pi_native_handle NativeHandle,
                                   bool KeepOwnership) {
  const auto &Plugin = pi::getPlugin<backend::level_zero>();
  // Create PI context first.
  pi_context PiContext;
  std::vector<pi_device> DeviceHandles;
  for (auto Dev : DeviceList) {
    DeviceHandles.push_back(detail::getSyclObjImpl(Dev)->getHandleRef());
  }
  Plugin.call<PiApiKind::piextContextCreateWithNativeHandle>(
      NativeHandle, DeviceHandles.size(), DeviceHandles.data(), !KeepOwnership,
      &PiContext);
  // Construct the SYCL context from PI context.
  return detail::createSyclObjFromImpl<context>(
      std::make_shared<context_impl>(PiContext, async_handler{}, Plugin));
}

// TODO: remove this version (without ownership) when allowed to break ABI.
__SYCL_EXPORT context make_context(const std::vector<device> &DeviceList,
                                   pi_native_handle NativeHandle) {
  return make_context(DeviceList, NativeHandle, false);
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
                               pi_native_handle NativeHandle,
                               bool KeepOwnership) {
  const auto &ContextImpl = getSyclObjImpl(Context);
  return detail::make_queue(NativeHandle, Context, KeepOwnership,
                            ContextImpl->get_async_handler(),
                            backend::level_zero);
}

// TODO: remove this version (without ownership) when allowed to break ABI.
__SYCL_EXPORT queue make_queue(const context &Context,
                               pi_native_handle NativeHandle) {
  return make_queue(Context, NativeHandle, false);
}

//----------------------------------------------------------------------------
// Implementation of level_zero::make<event>
__SYCL_EXPORT event make_event(const context &Context,
                               pi_native_handle NativeHandle,
                               bool KeepOwnership) {
  return detail::make_event(NativeHandle, Context, KeepOwnership,
                            backend::level_zero);
}

} // namespace level_zero
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
