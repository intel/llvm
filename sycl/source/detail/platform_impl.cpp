//==----------- platform_impl.cpp ------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/device_impl.hpp>
#include <CL/sycl/detail/platform_impl.hpp>
#include <CL/sycl/device.hpp>
#include <detail/config.hpp>

#include <algorithm>
#include <cstring>
#include <regex>

namespace cl {
namespace sycl {
namespace detail {

vector_class<platform> platform_impl::get_platforms() {
  vector_class<platform> Platforms;

  pi_uint32 NumPlatforms = 0;
  PI_CALL(piPlatformsGet)(0, nullptr, &NumPlatforms);
  info::device_type ForcedType = detail::get_forced_type();

  if (NumPlatforms) {
    vector_class<RT::PiPlatform> pi_platforms(NumPlatforms);
    PI_CALL(piPlatformsGet)(NumPlatforms, pi_platforms.data(), nullptr);

    for (pi_uint32 i = 0; i < NumPlatforms; i++) {

      platform plt = detail::createSyclObjFromImpl<platform>(
          std::make_shared<platform_impl>(pi_platforms[i]));
      // Skip platforms which do not contain requested device types
      if (!plt.get_devices(ForcedType).empty())
        Platforms.push_back(plt);
    }
  }
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

static void filterWhiteList(vector_class<RT::PiDevice> &pi_devices,
                            RT::PiPlatform pi_platform) {
  const std::vector<DevDescT> whiteList(getWhiteListDesc());
  if (whiteList.empty())
    return;

  const string_class platformName =
      sycl::detail::get_platform_info<string_class, info::platform::name>::get(
          pi_platform);

  const string_class platformVer = sycl::detail::get_platform_info<
      string_class, info::platform::version>::get(pi_platform);

  int insertIDx = 0;
  for (RT::PiDevice dev : pi_devices) {
    const string_class devName =
        sycl::detail::get_device_info<string_class, info::device::name>::get(
            dev);

    const string_class devDriverVer =
        sycl::detail::get_device_info<string_class,
                                      info::device::driver_version>::get(dev);

    for (const DevDescT &desc : whiteList) {
      if (nullptr != desc.platformName &&
          !std::regex_match(platformName,
                            std::regex(std::string(desc.platformName,
                                                   desc.platformNameSize))))
        continue;

      if (nullptr != desc.platformVer &&
          !std::regex_match(
              platformVer,
              std::regex(std::string(desc.platformVer, desc.platformVerSize))))
        continue;

      if (nullptr != desc.devName &&
          !std::regex_match(
              devName, std::regex(std::string(desc.devName, desc.devNameSize))))
        continue;

      if (nullptr != desc.devDriverVer &&
          !std::regex_match(devDriverVer,
                            std::regex(std::string(desc.devDriverVer,
                                                   desc.devDriverVerSize))))
        continue;

      pi_devices[insertIDx++] = dev;
      break;
    }
  }
  pi_devices.resize(insertIDx);
}

vector_class<device>
platform_impl::get_devices(info::device_type DeviceType) const {
  vector_class<device> Res;
  if (is_host() && (DeviceType == info::device_type::host ||
                    DeviceType == info::device_type::all)) {
    Res.resize(1); // default device construct creates host device
    return Res;
  }

  pi_uint32 NumDevices;
  PI_CALL(piDevicesGet)
  (MPlatform, pi::cast<RT::PiDeviceType>(DeviceType), 0,
   pi::cast<RT::PiDevice *>(nullptr), &NumDevices);

  if (NumDevices == 0)
    return Res;

  vector_class<RT::PiDevice> pi_devices(NumDevices);
  // TODO catch an exception and put it to list of asynchronous exceptions
  PI_CALL(piDevicesGet)
  (MPlatform, pi::cast<RT::PiDeviceType>(DeviceType), NumDevices,
   pi_devices.data(), nullptr);

  // Filter out devices that are not present in the white list
  if (SYCLConfig<SYCL_DEVICE_WHITE_LIST>::get())
    filterWhiteList(pi_devices, m_platform);

  std::for_each(pi_devices.begin(), pi_devices.end(),
                [&Res](const RT::PiDevice &a_pi_device) {
                  device sycl_device = detail::createSyclObjFromImpl<device>(
                      std::make_shared<device_impl>(a_pi_device));
                  Res.push_back(sycl_device);
                });
  return Res;
}
} // namespace detail
} // namespace sycl
} // namespace cl
