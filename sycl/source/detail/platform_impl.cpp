//==----------- platform_impl.cpp ------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/device_impl.hpp>
#include <CL/sycl/detail/platform_impl.hpp>
#include <CL/sycl/detail/platform_info.hpp>
#include <CL/sycl/device.hpp>
#include <detail/config.hpp>

#include <algorithm>
#include <cstring>
#include <regex>

__SYCL_INLINE namespace cl {
namespace sycl {
namespace detail {

vector_class<platform> platform_impl::get_platforms() {
  vector_class<platform> Platforms;

  pi_uint32 NumPlatforms = 0;
  PI_CALL(piPlatformsGet)(0, nullptr, &NumPlatforms);
  info::device_type ForcedType = detail::get_forced_type();

  if (NumPlatforms) {
    vector_class<RT::PiPlatform> PiPlatforms(NumPlatforms);
    PI_CALL(piPlatformsGet)(NumPlatforms, PiPlatforms.data(), nullptr);

    for (const auto &PiPlatform : PiPlatforms) {
      platform Platform = detail::createSyclObjFromImpl<platform>(
          std::make_shared<platform_impl>(PiPlatform));
      // Skip platforms which do not contain requested device
      // types
      if (!Platform.get_devices(ForcedType).empty())
        Platforms.push_back(Platform);
    }
  }

  // The host platform should always be available.
  Platforms.emplace_back(platform());

  return Platforms;
}

struct DevDescT {
  const char *devName = nullptr;
  int devNameSize = 0;

  const char *devDriverVer = nullptr;
  int devDriverVerSize = 0;

  const char *platformName = nullptr;
  int platformNameSize = 0;

  const char *platformVer = nullptr;
  int platformVerSize = 0;
};

static std::vector<DevDescT> getWhiteListDesc() {
  const char *str = SYCLConfig<SYCL_DEVICE_WHITE_LIST>::get();
  if (!str)
    return {};

  std::vector<DevDescT> decDescs;
  const char devNameStr[] = "DeviceName";
  const char driverVerStr[] = "DriverVersion";
  const char platformNameStr[] = "PlatformName";
  const char platformVerStr[] = "PlatformVersion";
  decDescs.emplace_back();
  while ('\0' != *str) {
    const char **valuePtr = nullptr;
    int *size = nullptr;

    // -1 to avoid comparing null terminator
    if (0 == strncmp(devNameStr, str, sizeof(devNameStr) - 1)) {
      valuePtr = &decDescs.back().devName;
      size = &decDescs.back().devNameSize;
      str += sizeof(devNameStr) - 1;
    } else if (0 ==
               strncmp(platformNameStr, str, sizeof(platformNameStr) - 1)) {
      valuePtr = &decDescs.back().platformName;
      size = &decDescs.back().platformNameSize;
      str += sizeof(platformNameStr) - 1;
    } else if (0 == strncmp(platformVerStr, str, sizeof(platformVerStr) - 1)) {
      valuePtr = &decDescs.back().platformVer;
      size = &decDescs.back().platformVerSize;
      str += sizeof(platformVerStr) - 1;
    } else if (0 == strncmp(driverVerStr, str, sizeof(driverVerStr) - 1)) {
      valuePtr = &decDescs.back().devDriverVer;
      size = &decDescs.back().devDriverVerSize;
      str += sizeof(driverVerStr) - 1;
    }

    if (':' != *str)
      throw sycl::runtime_error("Malformed device white list");

    // Skip ':'
    str += 1;

    if ('{' != *str || '{' != *(str + 1))
      throw sycl::runtime_error("Malformed device white list");

    // Skip opening sequence "{{"
    str += 2;

    *valuePtr = str;

    // Increment until closing sequence is encountered
    while (('\0' != *str) && ('}' != *str || '}' != *(str + 1)))
      ++str;

    if ('\0' == *str)
      throw sycl::runtime_error("Malformed device white list");

    *size = str - *valuePtr;

    // Skip closing sequence "}}"
    str += 2;

    if ('\0' == *str)
      break;

    // '|' means that the is another filter
    if ('|' == *str)
      decDescs.emplace_back();
    else if (',' != *str)
      throw sycl::runtime_error("Malformed device white list");

    ++str;
  }

  return decDescs;
}

static void filterWhiteList(vector_class<RT::PiDevice> &PiDevices,
                            RT::PiPlatform PiPlatform) {
  const std::vector<DevDescT> WhiteList(getWhiteListDesc());
  if (WhiteList.empty())
    return;

  const string_class PlatformName =
      sycl::detail::get_platform_info<string_class, info::platform::name>::get(
          PiPlatform);

  const string_class PlatformVer =
      sycl::detail::get_platform_info<string_class,
                                      info::platform::version>::get(PiPlatform);

  int InsertIDx = 0;
  for (RT::PiDevice Device : PiDevices) {
    const string_class DeviceName =
        sycl::detail::get_device_info<string_class, info::device::name>::get(
            Device);

    const string_class DeviceDriverVer = sycl::detail::get_device_info<
        string_class, info::device::driver_version>::get(Device);

    for (const DevDescT &Desc : WhiteList) {
      if (nullptr != Desc.platformName &&
          !std::regex_match(PlatformName,
                            std::regex(std::string(Desc.platformName,
                                                   Desc.platformNameSize))))
        continue;

      if (nullptr != Desc.platformVer &&
          !std::regex_match(
              PlatformVer,
              std::regex(std::string(Desc.platformVer, Desc.platformVerSize))))
        continue;

      if (nullptr != Desc.devName &&
          !std::regex_match(DeviceName, std::regex(std::string(
                                            Desc.devName, Desc.devNameSize))))
        continue;

      if (nullptr != Desc.devDriverVer &&
          !std::regex_match(DeviceDriverVer,
                            std::regex(std::string(Desc.devDriverVer,
                                                   Desc.devDriverVerSize))))
        continue;

      PiDevices[InsertIDx++] = Device;
      break;
    }
  }
  PiDevices.resize(InsertIDx);
}

vector_class<device>
platform_impl::get_devices(info::device_type DeviceType) const {
  vector_class<device> Res;
  if (is_host() && (DeviceType == info::device_type::host ||
                    DeviceType == info::device_type::all)) {
    Res.resize(1); // default device constructor creates host device
  }

  // If any DeviceType other than host was requested for host platform,
  // an empty vector will be returned.
  if (is_host() || DeviceType == info::device_type::host)
    return Res;

  pi_uint32 NumDevices;
  PI_CALL(piDevicesGet)(MPlatform, pi::cast<RT::PiDeviceType>(DeviceType), 0,
                        pi::cast<RT::PiDevice *>(nullptr), &NumDevices);

  if (NumDevices == 0)
    return Res;

  vector_class<RT::PiDevice> PiDevices(NumDevices);
  // TODO catch an exception and put it to list of asynchronous exceptions
  PI_CALL(piDevicesGet)(MPlatform, pi::cast<RT::PiDeviceType>(DeviceType),
                        NumDevices, PiDevices.data(), nullptr);

  // Filter out devices that are not present in the white list
  if (SYCLConfig<SYCL_DEVICE_WHITE_LIST>::get())
    filterWhiteList(PiDevices, MPlatform);

  std::transform(PiDevices.begin(), PiDevices.end(), std::back_inserter(Res),
                 [](const RT::PiDevice &PiDevice) -> device {
                   return detail::createSyclObjFromImpl<device>(
                       std::make_shared<device_impl>(PiDevice));
                 });

  return Res;
}

bool platform_impl::has_extension(const string_class &ExtensionName) const {
  if (is_host())
    return false;

  string_class AllExtensionNames =
      get_platform_info<string_class, info::platform::extensions>::get(
          MPlatform);
  return (AllExtensionNames.find(ExtensionName) != std::string::npos);
}

template <info::platform param>
typename info::param_traits<info::platform, param>::return_type
platform_impl::get_info() const {
  if (is_host())
    return get_platform_info_host<param>();

  return get_platform_info<
      typename info::param_traits<info::platform, param>::return_type,
      param>::get(this->getHandleRef());
}

#define PARAM_TRAITS_SPEC(param_type, param, ret_type)                         \
  template ret_type platform_impl::get_info<info::param_type::param>() const;

#include <CL/sycl/info/platform_traits.def>
#undef PARAM_TRAITS_SPEC

} // namespace detail
} // namespace sycl
} // namespace cl
