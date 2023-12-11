//==--------- level_zero.cpp - SYCL Level-Zero backend ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/platform_impl.hpp>
#include <detail/plugin.hpp>
#include <detail/program_impl.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/backend.hpp>
#include <sycl/sycl.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::level_zero {
using namespace detail;

//----------------------------------------------------------------------------
// Implementation of level_zero::make<platform>
__SYCL_EXPORT platform make_platform(pi_native_handle NativeHandle) {
  return detail::make_platform(NativeHandle, backend::ext_oneapi_level_zero);
}

//----------------------------------------------------------------------------
// Implementation of level_zero::make<device>
__SYCL_EXPORT device make_device(const platform &Platform,
                                 pi_native_handle NativeHandle) {
  const auto &Plugin = pi::getPlugin<backend::ext_oneapi_level_zero>();
  const auto &PlatformImpl = getSyclObjImpl(Platform);
  // Create PI device first.
  pi::PiDevice PiDevice;
  Plugin->call<PiApiKind::piextDeviceCreateWithNativeHandle>(
      NativeHandle, PlatformImpl->getHandleRef(), &PiDevice);

  return detail::createSyclObjFromImpl<device>(
      PlatformImpl->getOrMakeDeviceImpl(PiDevice, PlatformImpl));
}

//----------------------------------------------------------------------------
// Implementation of level_zero::make<context>
__SYCL_EXPORT context make_context(const std::vector<device> &DeviceList,
                                   pi_native_handle NativeHandle,
                                   bool KeepOwnership) {
  const auto &Plugin = pi::getPlugin<backend::ext_oneapi_level_zero>();
  // Create PI context first.
  pi_context PiContext;
  std::vector<pi_device> DeviceHandles;
  for (auto Dev : DeviceList) {
    DeviceHandles.push_back(detail::getSyclObjImpl(Dev)->getHandleRef());
  }
  Plugin->call<PiApiKind::piextContextCreateWithNativeHandle>(
      NativeHandle, DeviceHandles.size(), DeviceHandles.data(), !KeepOwnership,
      &PiContext);
  // Construct the SYCL context from PI context.
  return detail::createSyclObjFromImpl<context>(
      std::make_shared<context_impl>(PiContext, detail::defaultAsyncHandler,
                                     Plugin, DeviceList, !KeepOwnership));
}

//----------------------------------------------------------------------------
// Implementation of level_zero::make<queue>
__SYCL_EXPORT queue make_queue(const context &Context, const device &Device,
                               pi_native_handle NativeHandle, bool IsImmCmdList,
                               bool KeepOwnership,
                               const property_list &Properties) {
  const auto &ContextImpl = getSyclObjImpl(Context);
  return detail::make_queue(
      NativeHandle, IsImmCmdList, Context, &Device, KeepOwnership, Properties,
      ContextImpl->get_async_handler(), backend::ext_oneapi_level_zero);
}

//----------------------------------------------------------------------------
// Implementation of level_zero::make<event>
__SYCL_EXPORT event make_event(const context &Context,
                               pi_native_handle NativeHandle,
                               bool KeepOwnership) {
  return detail::make_event(NativeHandle, Context, KeepOwnership,
                            backend::ext_oneapi_level_zero);
}

} // namespace ext::oneapi::level_zero
} // namespace _V1
} // namespace sycl
