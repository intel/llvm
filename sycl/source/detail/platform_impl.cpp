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

    std::vector<PlatformImplPtr> &PlatformCache =
        GlobalHandler::instance().getPlatformCache();

    // If we've already seen this platform, return the impl
    for (const auto &PlatImpl : PlatformCache) {
      if (PlatImpl->getHandleRef() == PiPlatform)
        return PlatImpl;
    }

    // Otherwise make the impl
    Result = std::make_shared<platform_impl>(PiPlatform, Plugin);
    PlatformCache.emplace_back(Result);
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

static bool IsBannedPlatform(platform Platform) {
  // The NVIDIA OpenCL platform is currently not compatible with DPC++
  // since it is only 1.2 but gets selected by default in many systems
  // There is also no support on the PTX backend for OpenCL consumption,
  // and there have been some internal reports.
  // To avoid problems on default users and deployment of DPC++ on platforms
  // where CUDA is available, the OpenCL support is disabled.
  //
  auto IsNVIDIAOpenCL = [](platform Platform) {
    if (Platform.is_host())
      return false;

    const bool HasCUDA = Platform.get_info<info::platform::name>().find(
                             "NVIDIA CUDA") != std::string::npos;
    const auto Backend =
        detail::getSyclObjImpl(Platform)->getPlugin().getBackend();
    const bool IsCUDAOCL = (HasCUDA && Backend == backend::opencl);
    if (detail::pi::trace(detail::pi::TraceLevel::PI_TRACE_ALL) && IsCUDAOCL) {
      std::cout << "SYCL_PI_TRACE[all]: "
                << "NVIDIA CUDA OpenCL platform found but is not compatible."
                << std::endl;
    }
    return IsCUDAOCL;
  };
  return IsNVIDIAOpenCL(Platform);
}

vector_class<platform> platform_impl::get_platforms() {
  vector_class<platform> Platforms;
  const vector_class<plugin> &Plugins = RT::initialize();

  info::device_type ForcedType = detail::get_forced_type();
  for (unsigned int i = 0; i < Plugins.size(); i++) {

    pi_uint32 NumPlatforms = 0;
    // Move to the next plugin if the plugin fails to initialize.
    // This way platforms from other plugins get a chance to be discovered.
    if (Plugins[i].call_nocheck<PiApiKind::piPlatformsGet>(
            0, nullptr, &NumPlatforms) != PI_SUCCESS)
      continue;

    if (NumPlatforms) {
      vector_class<RT::PiPlatform> PiPlatforms(NumPlatforms);
      if (Plugins[i].call_nocheck<PiApiKind::piPlatformsGet>(
              NumPlatforms, PiPlatforms.data(), nullptr) != PI_SUCCESS)
        return Platforms;

      for (const auto &PiPlatform : PiPlatforms) {
        platform Platform = detail::createSyclObjFromImpl<platform>(
            getOrMakePlatformImpl(PiPlatform, Plugins[i]));
        // Skip platforms which do not contain requested device types
        if (!Platform.get_devices(ForcedType).empty() &&
            !IsBannedPlatform(Platform))
          Platforms.push_back(Platform);
      }
    }
  }

  // The host platform should always be available unless not allowed by the
  // SYCL_DEVICE_FILTER
  detail::device_filter_list *FilterList =
      detail::SYCLConfig<detail::SYCL_DEVICE_FILTER>::get();
  if (!FilterList || FilterList->backendCompatible(backend::host))
    Platforms.emplace_back(platform());

  return Platforms;
}

// Filter out the devices that are not compatible with SYCL_DEVICE_FILTER.
// All three entries (backend:device_type:device_num) are optional.
// The missing entries are constructed using '*', which means 'any' | 'all'
// by the device_filter constructor.
// This function matches devices in the order of backend, device_type, and
// device_num.
static void filterDeviceFilter(vector_class<RT::PiDevice> &PiDevices,
                               const plugin &Plugin) {
  device_filter_list *FilterList = SYCLConfig<SYCL_DEVICE_FILTER>::get();
  if (!FilterList)
    return;

  backend Backend = Plugin.getBackend();
  int InsertIDx = 0;
  int DeviceNum = 0;
  for (RT::PiDevice Device : PiDevices) {
    RT::PiDeviceType PiDevType;
    Plugin.call<PiApiKind::piDeviceGetInfo>(Device, PI_DEVICE_INFO_TYPE,
                                            sizeof(RT::PiDeviceType),
                                            &PiDevType, nullptr);
    // Assumption here is that there is 1-to-1 mapping between PiDevType and
    // Sycl device type for GPU, CPU, and ACC.
    info::device_type DeviceType = pi::cast<info::device_type>(PiDevType);

    for (const device_filter &Filter : FilterList->get()) {
      backend FilterBackend = Filter.Backend;
      // First, match the backend entry
      if (FilterBackend == Backend || FilterBackend == backend::all) {
        info::device_type FilterDevType = Filter.DeviceType;
        // Next, match the device_type entry
        if (FilterDevType == info::device_type::all) {
          // Last, match the device_num entry
          if (!Filter.HasDeviceNum || DeviceNum == Filter.DeviceNum) {
            PiDevices[InsertIDx++] = Device;
            break;
          }
        } else if (FilterDevType == DeviceType) {
          if (!Filter.HasDeviceNum || DeviceNum == Filter.DeviceNum) {
            PiDevices[InsertIDx++] = Device;
            break;
          }
        }
      }
    }
    DeviceNum++;
  }
  PiDevices.resize(InsertIDx);
}

std::shared_ptr<device_impl> platform_impl::getOrMakeDeviceImpl(
    RT::PiDevice PiDevice, const std::shared_ptr<platform_impl> &PlatformImpl) {
  const std::lock_guard<std::mutex> Guard(MDeviceMapMutex);

  // If we've already seen this device, return the impl
  for (const std::weak_ptr<device_impl> &DeviceWP : MDeviceCache) {
    if (std::shared_ptr<device_impl> Device = DeviceWP.lock()) {
      if (Device->getHandleRef() == PiDevice)
        return Device;
    }
  }

  // Otherwise make the impl
  std::shared_ptr<device_impl> Result =
      std::make_shared<device_impl>(PiDevice, PlatformImpl);
  MDeviceCache.emplace_back(Result);

  return Result;
}

vector_class<device>
platform_impl::get_devices(info::device_type DeviceType) const {
  vector_class<device> Res;
  if (is_host() && (DeviceType == info::device_type::host ||
                    DeviceType == info::device_type::all)) {
    // If SYCL_DEVICE_FILTER is set, check if filter contains host.
    device_filter_list *FilterList = SYCLConfig<SYCL_DEVICE_FILTER>::get();
    if (!FilterList || FilterList->containsHost()) {
      Res.push_back(device());
    }
  }

  // If any DeviceType other than host was requested for host platform,
  // an empty vector will be returned.
  if (is_host() || DeviceType == info::device_type::host)
    return Res;

  pi_uint32 NumDevices = 0;
  const detail::plugin &Plugin = getPlugin();
  Plugin.call<PiApiKind::piDevicesGet>(
      MPlatform, pi::cast<RT::PiDeviceType>(DeviceType), 0,
      pi::cast<RT::PiDevice *>(nullptr), &NumDevices);

  if (NumDevices == 0)
    return Res;

  vector_class<RT::PiDevice> PiDevices(NumDevices);
  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin.call<PiApiKind::piDevicesGet>(MPlatform,
                                       pi::cast<RT::PiDeviceType>(DeviceType),
                                       NumDevices, PiDevices.data(), nullptr);

  // Filter out devices that are not present in the SYCL_DEVICE_ALLOWLIST
  if (SYCLConfig<SYCL_DEVICE_ALLOWLIST>::get())
    applyAllowList(PiDevices, MPlatform, this->getPlugin());

  // Filter out devices that are not compatible with SYCL_DEVICE_FILTER
  filterDeviceFilter(PiDevices, Plugin);

  PlatformImplPtr PlatformImpl = getOrMakePlatformImpl(MPlatform, *MPlugin);
  std::transform(
      PiDevices.begin(), PiDevices.end(), std::back_inserter(Res),
      [PlatformImpl](const RT::PiDevice &PiDevice) -> device {
        return detail::createSyclObjFromImpl<device>(
            PlatformImpl->getOrMakeDeviceImpl(PiDevice, PlatformImpl));
      });

  return Res;
}

bool platform_impl::has_extension(const string_class &ExtensionName) const {
  if (is_host())
    return false;

  string_class AllExtensionNames =
      get_platform_info<string_class, info::platform::extensions>::get(
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
