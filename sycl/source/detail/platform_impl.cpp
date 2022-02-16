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
#include <mutex>
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

std::vector<platform> platform_impl::get_platforms() {
  std::vector<platform> Platforms;
  std::vector<plugin> &Plugins = RT::initialize();
  info::device_type ForcedType = detail::get_forced_type();
  for (plugin &Plugin : Plugins) {
    pi_uint32 NumPlatforms = 0;
    // Move to the next plugin if the plugin fails to initialize.
    // This way platforms from other plugins get a chance to be discovered.
    if (Plugin.call_nocheck<PiApiKind::piPlatformsGet>(
            0, nullptr, &NumPlatforms) != PI_SUCCESS)
      continue;

    if (NumPlatforms) {
      std::vector<RT::PiPlatform> PiPlatforms(NumPlatforms);
      if (Plugin.call_nocheck<PiApiKind::piPlatformsGet>(
              NumPlatforms, PiPlatforms.data(), nullptr) != PI_SUCCESS)
        return Platforms;

      for (const auto &PiPlatform : PiPlatforms) {
        platform Platform = detail::createSyclObjFromImpl<platform>(
            getOrMakePlatformImpl(PiPlatform, Plugin));
        {
          std::lock_guard<std::mutex> Guard(*Plugin.getPluginMutex());
          // insert PiPlatform into the Plugin
          Plugin.getPlatformId(PiPlatform);
        }
        // Skip platforms which do not contain requested device types
        if (!Platform.get_devices(ForcedType).empty() &&
            !IsBannedPlatform(Platform))
          Platforms.push_back(Platform);
      }
    }
  }

  // Register default context release handler after plugins have been loaded and
  // after the first calls to each plugin. This initializes a function-local
  // variable that should be destroyed before any global variables in the
  // plugins are destroyed. This is done after the first call to the backends to
  // ensure any lazy-loaded dependencies are loaded prior to the handler
  // variable's initialization. Note: The default context release handler is not
  // guaranteed to be destroyed before function-local static variables as they
  // may be initialized after.
  GlobalHandler::registerDefaultContextReleaseHandler();

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
static void filterDeviceFilter(std::vector<RT::PiDevice> &PiDevices,
                               RT::PiPlatform Platform) {
  device_filter_list *FilterList = SYCLConfig<SYCL_DEVICE_FILTER>::get();
  if (!FilterList)
    return;

  std::vector<plugin> &Plugins = RT::initialize();
  auto It =
      std::find_if(Plugins.begin(), Plugins.end(), [Platform](plugin &Plugin) {
        return Plugin.containsPiPlatform(Platform);
      });
  if (It == Plugins.end())
    return;

  plugin &Plugin = *It;
  backend Backend = Plugin.getBackend();
  int InsertIDx = 0;
  // DeviceIds should be given consecutive numbers across platforms in the same
  // backend
  std::lock_guard<std::mutex> Guard(*Plugin.getPluginMutex());
  int DeviceNum = Plugin.getStartingDeviceId(Platform);
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
  // remember the last backend that has gone through this filter function
  // to assign a unique device id number across platforms that belong to
  // the same backend. For example, opencl:cpu:0, opencl:acc:1, opencl:gpu:2
  Plugin.setLastDeviceId(Platform, DeviceNum);
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

std::vector<device>
platform_impl::get_devices(info::device_type DeviceType) const {
  std::vector<device> Res;
  if (is_host() && (DeviceType == info::device_type::host ||
                    DeviceType == info::device_type::all)) {
    // If SYCL_DEVICE_FILTER is set, check if filter contains host.
    device_filter_list *FilterList = SYCLConfig<SYCL_DEVICE_FILTER>::get();
    if (!FilterList || FilterList->containsHost()) {
      Res.push_back(device(host_device{}));
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

  std::vector<RT::PiDevice> PiDevices(NumDevices);
  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin.call<PiApiKind::piDevicesGet>(MPlatform,
                                       pi::cast<RT::PiDeviceType>(DeviceType),
                                       NumDevices, PiDevices.data(), nullptr);

  // Filter out devices that are not present in the SYCL_DEVICE_ALLOWLIST
  if (SYCLConfig<SYCL_DEVICE_ALLOWLIST>::get())
    applyAllowList(PiDevices, MPlatform, Plugin);

  // Filter out devices that are not compatible with SYCL_DEVICE_FILTER
  filterDeviceFilter(PiDevices, MPlatform);

  PlatformImplPtr PlatformImpl = getOrMakePlatformImpl(MPlatform, Plugin);
  std::transform(
      PiDevices.begin(), PiDevices.end(), std::back_inserter(Res),
      [PlatformImpl](const RT::PiDevice &PiDevice) -> device {
        return detail::createSyclObjFromImpl<device>(
            PlatformImpl->getOrMakeDeviceImpl(PiDevice, PlatformImpl));
      });

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
