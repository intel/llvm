//==----------- platform_impl.cpp ------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/device.hpp>
#include <detail/config.hpp>
#include <detail/device_impl.hpp>
#include <detail/force_device.hpp>
#include <detail/platform_impl.hpp>
#include <detail/platform_info.hpp>

#include <algorithm>
#include <cstring>
#include <regex>
#include <string>

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
  static std::vector<PlatformImplPtr> PlatformCache;
  static std::mutex PlatformMapMutex;

  PlatformImplPtr Result;
  {
    const std::lock_guard<std::mutex> Guard(PlatformMapMutex);

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
    Plugins[i].call<PiApiKind::piPlatformsGet>(0, nullptr, &NumPlatforms);

    if (NumPlatforms) {
      vector_class<RT::PiPlatform> PiPlatforms(NumPlatforms);
      Plugins[i].call<PiApiKind::piPlatformsGet>(NumPlatforms,
                                                 PiPlatforms.data(), nullptr);

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

  // The host platform should always be available.
  Platforms.emplace_back(platform());

  return Platforms;
}

struct DevDescT {
  std::string devName;
  std::string devDriverVer;
  std::string platName;
  std::string platVer;
};

static std::vector<DevDescT> getAllowListDesc() {
  std::string allowList(SYCLConfig<SYCL_DEVICE_ALLOWLIST>::get());
  if (allowList.empty())
    return {};

  std::string deviceName("DeviceName:");
  std::string driverVersion("DriverVersion:");
  std::string platformName("PlatformName:");
  std::string platformVersion("PlatformVersion:");
  std::vector<DevDescT> decDescs;
  decDescs.emplace_back();

  size_t pos = 0;
  size_t prev = pos;
  while (pos < allowList.size()) {
    if ((allowList.compare(pos, deviceName.size(), deviceName)) == 0) {
      prev = pos;
      if ((pos = allowList.find("{{", pos)) == std::string::npos) {
        throw sycl::runtime_error("Malformed syntax in SYCL_DEVICE_ALLOWLIST",
                                  PI_INVALID_VALUE);
      }
      if (pos > prev + deviceName.size()) {
        throw sycl::runtime_error("Malformed syntax in SYCL_DEVICE_ALLOWLIST",
                                  PI_INVALID_VALUE);
      }

      pos = pos + 2;
      size_t start = pos;
      if ((pos = allowList.find("}}", pos)) == std::string::npos) {
        throw sycl::runtime_error("Malformed syntax in SYCL_DEVICE_ALLOWLIST",
                                  PI_INVALID_VALUE);
      }
      decDescs.back().devName = allowList.substr(start, pos - start);
      pos = pos + 2;

      if (allowList[pos] == ',') {
        pos++;
      }
    }

    else if ((allowList.compare(pos, driverVersion.size(), driverVersion)) ==
             0) {
      prev = pos;
      if ((pos = allowList.find("{{", pos)) == std::string::npos) {
        throw sycl::runtime_error("Malformed syntax in SYCL_DEVICE_ALLOWLIST",
                                  PI_INVALID_VALUE);
      }
      if (pos > prev + driverVersion.size()) {
        throw sycl::runtime_error("Malformed syntax in SYCL_DEVICE_ALLOWLIST",
                                  PI_INVALID_VALUE);
      }

      size_t start = pos + 2;
      if ((pos = allowList.find("}}", pos)) == std::string::npos) {
        throw sycl::runtime_error("Malformed syntax in SYCL_DEVICE_ALLOWLIST",
                                  PI_INVALID_VALUE);
      }
      decDescs.back().devDriverVer = allowList.substr(start, pos - start);
      pos = pos + 2;

      if (allowList[pos] == ',') {
        pos++;
      }
    }

    else if ((allowList.compare(pos, platformName.size(), platformName)) == 0) {
      prev = pos;
      if ((pos = allowList.find("{{", pos)) == std::string::npos) {
        throw sycl::runtime_error("Malformed syntax in SYCL_DEVICE_ALLOWLIST",
                                  PI_INVALID_VALUE);
      }
      if (pos > prev + platformName.size()) {
        throw sycl::runtime_error("Malformed syntax in SYCL_DEVICE_ALLOWLIST",
                                  PI_INVALID_VALUE);
      }

      size_t start = pos + 2;
      if ((pos = allowList.find("}}", pos)) == std::string::npos) {
        throw sycl::runtime_error("Malformed syntax in SYCL_DEVICE_ALLOWLIST",
                                  PI_INVALID_VALUE);
      }
      decDescs.back().platName = allowList.substr(start, pos - start);
      pos = pos + 2;

      if (allowList[pos] == ',') {
        pos++;
      }

    }

    else if ((allowList.compare(pos, platformVersion.size(),
                                platformVersion)) == 0) {
      prev = pos;
      if ((pos = allowList.find("{{", pos)) == std::string::npos) {
        throw sycl::runtime_error("Malformed syntax in SYCL_DEVICE_ALLOWLIST",
                                  PI_INVALID_VALUE);
      }
      if (pos > prev + platformVersion.size()) {
        throw sycl::runtime_error("Malformed syntax in SYCL_DEVICE_ALLOWLIST",
                                  PI_INVALID_VALUE);
      }

      size_t start = pos + 2;
      if ((pos = allowList.find("}}", pos)) == std::string::npos) {
        throw sycl::runtime_error("Malformed syntax in SYCL_DEVICE_ALLOWLIST",
                                  PI_INVALID_VALUE);
      }
      decDescs.back().platVer = allowList.substr(start, pos - start);
      pos = pos + 2;
    }

    else if (allowList.find('|', pos) != std::string::npos) {
      pos = allowList.find('|') + 1;
      while (allowList[pos] == ' ') {
        pos++;
      }
      decDescs.emplace_back();
    }

    else {
      throw sycl::runtime_error("Unrecognized key in device allowlist",
                                PI_INVALID_VALUE);
    }
  } // while (pos <= allowList.size())
  return decDescs;
}

std::vector<int> convertVersionString(std::string version) {
  // version string format is xx.yy.zzzzz
  std::vector<int> values;
  size_t pos = 0;
  size_t start = pos;
  if ((pos = version.find(".", pos)) == std::string::npos) {
    throw sycl::runtime_error("Malformed syntax in version string",
                              PI_INVALID_VALUE);
  }
  values.push_back(std::stoi(version.substr(start, pos)));
  pos++;
  start = pos;
  if ((pos = version.find(".", pos)) == std::string::npos) {
    throw sycl::runtime_error("Malformed syntax in version string",
                              PI_INVALID_VALUE);
  }
  values.push_back(std::stoi(version.substr(start, pos)));
  pos++;
  values.push_back(std::stoi(version.substr(pos)));

  return values;
}

enum MatchState { UNKNOWN, MATCH, NOMATCH };

MatchState matchVersions(std::string version1, std::string version2) {
  std::vector<int> v1 = convertVersionString(version1);
  std::vector<int> v2 = convertVersionString(version2);
  if (v1[0] >= v2[0] && v1[1] >= v2[1] && v1[2] >= v2[2]) {
    return MatchState::MATCH;
  } else {
    return MatchState::NOMATCH;
  }
}

static void filterAllowList(vector_class<RT::PiDevice> &PiDevices,
                            RT::PiPlatform PiPlatform, const plugin &Plugin) {
  const std::vector<DevDescT> AllowList(getAllowListDesc());
  if (AllowList.empty())
    return;

  MatchState devNameState = UNKNOWN;
  MatchState devVerState = UNKNOWN;
  MatchState platNameState = UNKNOWN;
  MatchState platVerState = UNKNOWN;

  const string_class PlatformName =
      sycl::detail::get_platform_info<string_class, info::platform::name>::get(
          PiPlatform, Plugin);

  const string_class PlatformVer =
      sycl::detail::get_platform_info<string_class,
                                      info::platform::version>::get(PiPlatform,
                                                                    Plugin);

  int InsertIDx = 0;
  for (RT::PiDevice Device : PiDevices) {
    const string_class DeviceName =
        sycl::detail::get_device_info<string_class, info::device::name>::get(
            Device, Plugin);

    const string_class DeviceDriverVer = sycl::detail::get_device_info<
        string_class, info::device::driver_version>::get(Device, Plugin);

    for (const DevDescT &Desc : AllowList) {
      if (!Desc.platName.empty()) {
        if (!std::regex_match(PlatformName, std::regex(Desc.platName))) {
          platNameState = MatchState::NOMATCH;
          continue;
        } else {
          platNameState = MatchState::MATCH;
        }
      }

      if (!Desc.platVer.empty()) {
        if (!std::regex_match(PlatformVer, std::regex(Desc.platVer))) {
          platVerState = MatchState::NOMATCH;
          continue;
        } else {
          platVerState = MatchState::MATCH;
        }
      }

      if (!Desc.devName.empty()) {
        if (!std::regex_match(DeviceName, std::regex(Desc.devName))) {
          devNameState = MatchState::NOMATCH;
          continue;
        } else {
          devNameState = MatchState::MATCH;
        }
      }

      if (!Desc.devDriverVer.empty()) {
        if (!std::regex_match(DeviceDriverVer, std::regex(Desc.devDriverVer))) {
          devVerState = matchVersions(DeviceDriverVer, Desc.devDriverVer);
          if (devVerState == MatchState::NOMATCH) {
            continue;
          }
        } else {
          devVerState = MatchState::MATCH;
        }
      }

      PiDevices[InsertIDx++] = Device;
      break;
    }
  }
  if (devNameState == MatchState::MATCH && devVerState == MatchState::NOMATCH) {
    throw sycl::runtime_error("Requested SYCL device not found",
                              PI_DEVICE_NOT_FOUND);
  }
  if (platNameState == MatchState::MATCH &&
      platVerState == MatchState::NOMATCH) {
    throw sycl::runtime_error("Requested SYCL platform not found",
                              PI_DEVICE_NOT_FOUND);
  }
  PiDevices.resize(InsertIDx);
}

std::shared_ptr<device_impl> platform_impl::getOrMakeDeviceImpl(
    RT::PiDevice PiDevice, const std::shared_ptr<platform_impl> &PlatformImpl) {
  const std::lock_guard<std::mutex> Guard(MDeviceMapMutex);

  // If we've already seen this device, return the impl
  for (const std::shared_ptr<device_impl> &Device : MDeviceCache) {
    if (Device->getHandleRef() == PiDevice)
      return Device;
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
    Res.push_back(device());
  }

  // If any DeviceType other than host was requested for host platform,
  // an empty vector will be returned.
  if (is_host() || DeviceType == info::device_type::host)
    return Res;

  pi_uint32 NumDevices;
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

  // Filter out devices that are not present in the allowlist
  if (SYCLConfig<SYCL_DEVICE_ALLOWLIST>::get())
    filterAllowList(PiDevices, MPlatform, this->getPlugin());

  PlatformImplPtr PlatformImpl = getOrMakePlatformImpl(MPlatform, *MPlugin);
  std::transform(
      PiDevices.begin(), PiDevices.end(), std::back_inserter(Res),
      [this, PlatformImpl](const RT::PiDevice &PiDevice) -> device {
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

#define PARAM_TRAITS_SPEC(param_type, param, ret_type)                         \
  template ret_type platform_impl::get_info<info::param_type::param>() const;

#include <CL/sycl/info/platform_traits.def>
#undef PARAM_TRAITS_SPEC

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
