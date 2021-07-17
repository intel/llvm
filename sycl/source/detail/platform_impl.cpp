//==----------- platform_impl.cpp ------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/device.hpp>
#include <detail/allowlist.hpp>
#include <detail/config.hpp>
#include <detail/device_impl.hpp>
#include <detail/force_device.hpp>
#include <detail/global_handler.hpp>
#include <detail/platform_impl.hpp>
#include <detail/platform_info.hpp>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

using PlatformImplPtr = std::shared_ptr<platform_impl>;

PlatformImplPtr platform_impl::getHostPlatformImpl() {
  static PlatformImplPtr HostImpl = std::make_shared<platform_impl>();
  return HostImpl;
}

PlatformImplPtr platform_impl::getOrMakePlatformImpl(RT::PiPlatform PiPlatform,
                                                     const plugin &Plugin) {
  PlatformImplPtr Result;
  {
    const std::lock_guard<std::mutex> Guard(
        GlobalHandler::instance().getPlatformMapMutex());

    std::map<PlatformImplPtr, std::vector<DeviceImplPtr>> &PlatformDeviceCache =
        GlobalHandler::instance().getPlatformDeviceCache();

    // If we've already seen this platform, return the impl
    for (const auto &Pair : PlatformDeviceCache) {
      if (Pair.first->getHandleRef() == PiPlatform)
        return Pair.first;
    }

    // Otherwise make the impl
    Result = std::make_shared<platform_impl>(PiPlatform, Plugin);
    PlatformDeviceCache[Result] = std::vector<DeviceImplPtr>();
  }

  return Result;
}

PlatformImplPtr platform_impl::getPlatformFromPiDevice(RT::PiDevice PiDevice,
                                                       const plugin &Plugin) {
  RT::PiPlatform Plt = nullptr; // TODO catch an exception and put it to list
  // of asynchronous exceptions
  Plugin.call<PiApiKind::piDeviceGetInfo>(PiDevice, PI_DEVICE_INFO_PLATFORM,
                                          sizeof(Plt), &Plt, nullptr);
  return getOrMakePlatformImpl(Plt, Plugin);
}

std::vector<platform> platform_impl::get_platforms() {
  std::vector<platform> Platforms;
  RT::initialize();
  const std::lock_guard<std::mutex> Guard(
      GlobalHandler::instance().getPlatformMapMutex());
  std::map<PlatformImplPtr, std::vector<DeviceImplPtr>> &PlatformDeviceCache =
      GlobalHandler::instance().getPlatformDeviceCache();
  for (const auto &Pair : PlatformDeviceCache) {
    platform Platform = detail::createSyclObjFromImpl<platform>(Pair.first);
    Platforms.push_back(Platform);
  }
  return Platforms;
}

DeviceImplPtr platform_impl::getOrMakeDeviceImpl(
    RT::PiDevice PiDevice, const std::shared_ptr<platform_impl> &PlatformImpl) {

  const std::lock_guard<std::mutex> Guard(
      GlobalHandler::instance().getPlatformMapMutex());
  std::map<PlatformImplPtr, std::vector<DeviceImplPtr>> &PlatformDeviceCache =
      GlobalHandler::instance().getPlatformDeviceCache();

  // If we've already seen this device, return the impl
  std::vector<DeviceImplPtr>& DeviceCache = PlatformDeviceCache[PlatformImpl];
  for (const DeviceImplPtr &Device : DeviceCache) {
    if (Device->getHandleRef() == PiDevice)
      return Device;
  }

  // Otherwise make the impl
  std::shared_ptr<device_impl> Result =
      std::make_shared<device_impl>(PiDevice, PlatformImpl);
  DeviceCache.emplace_back(Result);
  return Result;
}

std::vector<device>
platform_impl::get_devices(info::device_type DeviceType) const {
  std::vector<device> Res;
  const std::lock_guard<std::mutex> Guard(
      GlobalHandler::instance().getPlatformMapMutex());
  std::map<PlatformImplPtr, std::vector<DeviceImplPtr>> &PlatformDeviceCache =
      GlobalHandler::instance().getPlatformDeviceCache();

  // If we've already seen this device, return the impl
  PlatformImplPtr Platform = nullptr;
  for (const auto &Pair : PlatformDeviceCache) {
    if (Pair.first.get() == this) {
      Platform = Pair.first;
    }
  }
  std::vector<DeviceImplPtr>& DeviceCache = PlatformDeviceCache[Platform];
  for (const DeviceImplPtr &Device : DeviceCache) {
    // Assumption here is that there is 1-to-1 mapping between PiDevType and
    // Sycl device type for GPU, CPU, and ACC.
    info::device_type PiDeviceType =
        pi::cast<info::device_type>(Device->get_device_type());
    if (DeviceType == info::device_type::all || DeviceType == PiDeviceType)
      Res.push_back(detail::createSyclObjFromImpl<device>(Device));
  }

  return Res;
}

bool platform_impl::has_extension(const std::string &ExtensionName) const {
  if (is_host())
    return false;

  std::string AllExtensionNames =
      get_platform_info<std::string, info::platform::extensions>::get(
          MPlatform, getPlugin());
  return (AllExtensionNames.find(ExtensionName) != std::string::npos);
}

pi_native_handle platform_impl::getNative() const {
  const auto &Plugin = getPlugin();
  pi_native_handle Handle;
  Plugin.call<PiApiKind::piextPlatformGetNativeHandle>(getHandleRef(), &Handle);
  return Handle;
}

template <info::platform param>
typename info::param_traits<info::platform, param>::return_type
platform_impl::get_info() const {
  if (is_host())
    return get_platform_info_host<param>();

  return get_platform_info<
      typename info::param_traits<info::platform, param>::return_type,
      param>::get(this->getHandleRef(), getPlugin());
}

// All devices on the platform must have the given aspect.
bool platform_impl::has(aspect Aspect) const {
  for (const auto &dev : get_devices()) {
    if (dev.has(Aspect) == false) {
      return false;
    }
  }
  return true;
}

#define __SYCL_PARAM_TRAITS_SPEC(param_type, param, ret_type)                  \
  template ret_type platform_impl::get_info<info::param_type::param>() const;

#include <CL/sycl/info/platform_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
