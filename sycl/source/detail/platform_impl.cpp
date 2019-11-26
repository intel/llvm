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

#include <algorithm>
#include <cstring>
#include <regex>

namespace cl {
namespace sycl {
namespace detail {

vector_class<platform>
platform_impl_pi::get_platforms() {
  vector_class<platform> platforms;

  pi_uint32 num_platforms = 0;
  PI_CALL(piPlatformsGet)(0, nullptr, &num_platforms);
  info::device_type forced_type = detail::get_forced_type();

  if (num_platforms) {
    vector_class<RT::PiPlatform> pi_platforms(num_platforms);
    PI_CALL(piPlatformsGet)(num_platforms, pi_platforms.data(), nullptr);

    for (pi_uint32 i = 0; i < num_platforms; i++) {

      platform plt =
        detail::createSyclObjFromImpl<platform>(
          std::make_shared<platform_impl_pi>(pi_platforms[i]));
      // Skip platforms which do not contain requested device types
      if (!plt.get_devices(forced_type).empty())
        platforms.push_back(plt);
    }
  }
  return platforms;
}

vector_class<device>
platform_impl_host::get_devices(info::device_type dev_type) const {
  vector_class<device> res;
  if (dev_type == info::device_type::host || dev_type == info::device_type::all)
    res.resize(1); // default device construct creates host device
  return res;
}

struct DevDescT {
  const char *devName = nullptr;
  int devNameSize = 0;

  const char *devDriverVer = nullptr;
  int devDriverVerSize = 0;
};

static std::vector<DevDescT> getWhiteListDesc() {
  const char *str = SYCLConfig<SYCL_DEVICE_WHITE_LIST>::get();
  if (!str)
    return {};

  std::vector<DevDescT> decDescs;
  const char devNameStr[] = "DeviceName";
  const char driverVerStr[] = "DriverVersion";
  decDescs.emplace_back();
  while ('\0' != *str) {
    const char **valuePtr = nullptr;
    int *size = nullptr;

    // -1 to avoid comparing null terminator
    if (0 == strncmp(devNameStr, str, sizeof(devNameStr) - 1)) {
      valuePtr = &decDescs.back().devName;
      size = &decDescs.back().devNameSize;
      str += sizeof(devNameStr) - 1;
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

static void filterWhiteList(vector_class<RT::PiDevice> &pi_devices) {
  const std::vector<DevDescT> whiteList(getWhiteListDesc());
  if (whiteList.empty())
    return;

  int insertIDx = 0;
  for (RT::PiDevice dev : pi_devices) {
    const string_class devName =
        sycl::detail::get_device_info<string_class, info::device::name>::_(dev);

    const string_class devDriverVer =
        sycl::detail::get_device_info<string_class,
                                      info::device::driver_version>::_(dev);

    for (const DevDescT &desc : whiteList) {
      // At least device name is required field to consider the filter so far
      if (nullptr == desc.devName ||
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
platform_impl_pi::get_devices(info::device_type deviceType) const {
  vector_class<device> res;
  if (deviceType == info::device_type::host)
    return res;

  pi_uint32 num_devices;
  PI_CALL(piDevicesGet)(m_platform, pi::cast<RT::PiDeviceType>(deviceType), 0,
                        pi::cast<RT::PiDevice *>(nullptr), &num_devices);

  if (num_devices == 0)
    return res;

  vector_class<RT::PiDevice> pi_devices(num_devices);
  // TODO catch an exception and put it to list of asynchronous exceptions
  PI_CALL(piDevicesGet)(m_platform, pi::cast<RT::PiDeviceType>(deviceType),
                        num_devices, pi_devices.data(), nullptr);

  // Filter out devices that are not present in the white list
  if (SYCLConfig<SYCL_DEVICE_WHITE_LIST>::get())
    filterWhiteList(pi_devices);

  std::for_each(pi_devices.begin(), pi_devices.end(),
                [&res](const RT::PiDevice &a_pi_device) {
                  device sycl_device = detail::createSyclObjFromImpl<device>(
                      std::make_shared<device_impl>(a_pi_device));
                  res.push_back(sycl_device);
                });
  return res;
}

} // namespace detail
} // namespace sycl
} // namespace cl
