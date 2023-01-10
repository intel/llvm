// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef UR_CONFORMANCE_INCLUDE_UTILS_H_INCLUDED
#define UR_CONFORMANCE_INCLUDE_UTILS_H_INCLUDED

#include <string>
#include <uur/environment.h>

namespace uur {

/// @brief Make a string a valid identifier for gtest.
/// @param str The string to sanitize.
inline std::string GTestSanitizeString(const std::string &str) {
  auto str_cpy = str;
  std::replace_if(
      str_cpy.begin(), str_cpy.end(), [](char c) { return !std::isalnum(c); },
      '_');
  return str_cpy;
}

inline ur_platform_handle_t GetPlatform() {
  return PlatformEnvironment::instance->platform;
}

inline std::string GetPlatformName(ur_platform_handle_t hPlatform) {
  size_t name_len = 0;
  urPlatformGetInfo(hPlatform, UR_PLATFORM_INFO_NAME, 0, nullptr, &name_len);
  std::string platform_name(name_len, '\0');
  urPlatformGetInfo(hPlatform, UR_PLATFORM_INFO_NAME, name_len,
                    &platform_name[0], nullptr);
  platform_name.resize(platform_name.find_first_of('\0'));
  return GTestSanitizeString(
      std::string(platform_name.data(), platform_name.size()));
}

inline std::string GetDeviceName(ur_device_handle_t device) {
  size_t name_len = 0;
  urDeviceGetInfo(device, UR_DEVICE_INFO_NAME, 0, nullptr, &name_len);
  std::string name(name_len, '\0');
  urDeviceGetInfo(device, UR_DEVICE_INFO_NAME, name_len, &name[0], nullptr);
  name.resize(name.find_first_of('\0'));
  return GTestSanitizeString(std::string(name.data(), name.size()));
}

inline std::string GetPlatformAndDeviceName(ur_device_handle_t device) {
  return GetPlatformName(GetPlatform()) + "__" + GetDeviceName(device);
};

} // namespace uur

#endif // UR_CONFORMANCE_INCLUDE_UTILS_H_INCLUDED
