// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "ur_api.h"
#include "ur_print.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#define UR_CHECK(ACTION)                                                       \
  if (auto error = ACTION) {                                                   \
    std::cerr << "error: " #ACTION " failed: " << error << "\n";               \
    std::exit(1);                                                              \
  }                                                                            \
  (void)0

#define UR_CHECK_WEAK(ACTION)                                                  \
  if (auto error = ACTION) {                                                   \
    std::cout << error << "\n";                                                \
    return;                                                                    \
  }                                                                            \
  (void)0

namespace urinfo {
inline std::string stripPrefix(std::string_view value,
                               std::string_view prefix) {
  if (std::equal(prefix.begin(), prefix.end(), value.begin(),
                 value.begin() + std::min(value.size(), prefix.size()))) {
    value.remove_prefix(prefix.size());
  }
  return std::string(value);
}

inline std::string getAdapterBackend(ur_adapter_handle_t adapter) {
  ur_adapter_backend_t adapterBackend;
  UR_CHECK(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND,
                            sizeof(ur_adapter_backend_t), &adapterBackend,
                            nullptr));
  std::stringstream adapterBackendStream;
  adapterBackendStream << adapterBackend;
  std::string adapterBackendStr =
      stripPrefix(adapterBackendStream.str(), "UR_ADAPTER_BACKEND_");
  std::transform(adapterBackendStr.begin(), adapterBackendStr.end(),
                 adapterBackendStr.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return adapterBackendStr;
}

inline std::string getDeviceType(ur_device_handle_t device) {
  ur_device_type_t deviceType;
  UR_CHECK(urDeviceGetInfo(device, UR_DEVICE_INFO_TYPE,
                           sizeof(ur_device_type_t), &deviceType, nullptr));
  std::stringstream deviceTypeStream;
  deviceTypeStream << deviceType;
  std::string deviceTypeStr =
      stripPrefix(deviceTypeStream.str(), "UR_DEVICE_TYPE_");
  std::transform(deviceTypeStr.begin(), deviceTypeStr.end(),
                 deviceTypeStr.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return deviceTypeStr;
}

inline std::string getPlatformName(ur_platform_handle_t platform) {
  size_t nameSize = 0;
  UR_CHECK(urPlatformGetInfo(platform, UR_PLATFORM_INFO_NAME, 0, nullptr,
                             &nameSize));
  std::string name(nameSize, '\0');
  UR_CHECK(urPlatformGetInfo(platform, UR_PLATFORM_INFO_NAME, nameSize,
                             name.data(), &nameSize));
  name.pop_back(); // std::string does not need a terminating NULL, remove it
                   // here
  return name;
}

inline std::string getDeviceName(ur_device_handle_t device) {
  size_t nameSize = 0;
  UR_CHECK(urDeviceGetInfo(device, UR_DEVICE_INFO_NAME, 0, nullptr, &nameSize));
  std::string name(nameSize, '\0');
  UR_CHECK(urDeviceGetInfo(device, UR_DEVICE_INFO_NAME, nameSize, name.data(),
                           &nameSize));
  name.pop_back(); // std::string does not need a terminating NULL, remove it
                   // here
  return name;
}

inline std::string getDeviceVersion(ur_device_handle_t device) {
  size_t versionSize = 0;
  UR_CHECK(urDeviceGetInfo(device, UR_DEVICE_INFO_VERSION, 0, nullptr,
                           &versionSize));
  std::string name(versionSize, '\0');
  UR_CHECK(urDeviceGetInfo(device, UR_DEVICE_INFO_VERSION, versionSize,
                           name.data(), &versionSize));
  name.pop_back(); // std::string does not need a terminating NULL, remove it
                   // here
  return name;
}

inline std::string getDeviceDriverVersion(ur_device_handle_t device) {
  size_t driverVersionSize = 0;
  UR_CHECK(urDeviceGetInfo(device, UR_DEVICE_INFO_DRIVER_VERSION, 0, nullptr,
                           &driverVersionSize));
  std::string name(driverVersionSize, '\0');
  UR_CHECK(urDeviceGetInfo(device, UR_DEVICE_INFO_DRIVER_VERSION,
                           driverVersionSize, name.data(), &driverVersionSize));
  name.pop_back(); // std::string does not need a terminating NULL, remove it
                   // here
  return name;
}

inline std::string getLoaderConfigInfoName(ur_loader_config_info_t info) {
  std::stringstream stream;
  stream << info;
  return stripPrefix(stream.str(), "UR_LOADER_CONFIG_INFO_");
}

template <class T>
inline void printLoaderConfigInfo(ur_loader_config_handle_t loaderConfig,
                                  ur_loader_config_info_t info) {
  std::cout << getLoaderConfigInfoName(info) << ": ";
  T value;
  UR_CHECK(urLoaderConfigGetInfo(
      loaderConfig, info, sizeof(ur_adapter_backend_t), &value, nullptr));
  std::cout << value << "\n";
}

template <>
inline void
printLoaderConfigInfo<char[]>(ur_loader_config_handle_t loaderConfig,
                              ur_loader_config_info_t info) {
  std::cout << getLoaderConfigInfoName(info) << ": ";
  size_t size = 0;
  UR_CHECK_WEAK(urLoaderConfigGetInfo(loaderConfig, info, 0, nullptr, &size));
  std::string str(size, '\0');
  UR_CHECK_WEAK(
      urLoaderConfigGetInfo(loaderConfig, info, size, str.data(), nullptr));
  str.pop_back(); // std::string does not need a terminating NULL, remove it
                  // here
  std::cout << str << "\n";
}

inline std::string getAdapterInfoName(ur_adapter_info_t info) {
  std::stringstream stream;
  stream << info;
  return stripPrefix(stream.str(), "UR_ADAPTER_INFO_");
}

template <class T>
inline void printAdapterInfo(ur_adapter_handle_t adapter,
                             ur_adapter_info_t info) {
  std::cout << getAdapterInfoName(info) << ": ";
  T value;
  UR_CHECK(urAdapterGetInfo(adapter, info, sizeof(ur_adapter_backend_t), &value,
                            nullptr));
  std::cout << value << "\n";
}

inline std::string getPlatformInfoName(ur_platform_info_t info) {
  std::stringstream stream;
  stream << info;
  return stripPrefix(stream.str(), "UR_PLATFORM_INFO_");
}

template <class T>
inline void printPlatformInfo(ur_platform_handle_t platform,
                              ur_platform_info_t info) {
  std::cout << getPlatformInfoName(info) << ": ";
  T value;
  UR_CHECK_WEAK(urPlatformGetInfo(platform, info, sizeof(T), &value, nullptr));
  std::cout << value << "\n";
}

template <>
inline void printPlatformInfo<char[]>(ur_platform_handle_t platform,
                                      ur_platform_info_t info) {
  std::cout << getPlatformInfoName(info) << ": ";
  size_t size = 0;
  UR_CHECK_WEAK(urPlatformGetInfo(platform, info, 0, nullptr, &size));
  std::string str(size, '\0');
  UR_CHECK_WEAK(urPlatformGetInfo(platform, info, size, str.data(), nullptr));
  str.pop_back(); // std::string does not need a terminating NULL, remove it
                  // here
  std::cout << str << "\n";
}

inline std::string getDeviceInfoName(ur_device_info_t info) {
  std::stringstream stream;
  stream << info;
  return stripPrefix(stream.str(), "UR_DEVICE_INFO_");
}

template <class T>
inline std::enable_if_t<!std::is_array_v<T>>
printDeviceInfo(ur_device_handle_t device, ur_device_info_t info) {
  std::cout << getDeviceInfoName(info) << ": ";
  T value;
  UR_CHECK_WEAK(urDeviceGetInfo(device, info, sizeof(T), &value, nullptr));
  std::cout << value << "\n";
}

template <>
inline void printDeviceInfo<ur_bool_t>(ur_device_handle_t device,
                                       ur_device_info_t info) {
  std::cout << getDeviceInfoName(info) << ": ";
  ur_bool_t value;
  UR_CHECK_WEAK(
      urDeviceGetInfo(device, info, sizeof(ur_bool_t), &value, nullptr));
  std::string result = value ? "true" : "false";
  std::cout << result << "\n";
}

template <class T>
inline std::enable_if_t<std::is_array_v<T>>
printDeviceInfo(ur_device_handle_t device, ur_device_info_t info) {
  std::cout << getDeviceInfoName(info) << ": ";
  using value_t = std::remove_reference_t<decltype(std::declval<T>()[0])>;
  size_t size;
  UR_CHECK_WEAK(urDeviceGetInfo(device, info, 0, nullptr, &size));
  std::vector<value_t> values(size / sizeof(value_t));
  UR_CHECK_WEAK(urDeviceGetInfo(device, info, size, values.data(), nullptr));
  std::cout << "{ ";
  for (size_t i = 0; i < values.size(); i++) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << values[i];
  }
  std::cout << " }\n";
}

template <>
inline void printDeviceInfo<char[]>(ur_device_handle_t device,
                                    ur_device_info_t info) {
  std::cout << getDeviceInfoName(info) << ": ";
  size_t size = 0;
  UR_CHECK_WEAK(urDeviceGetInfo(device, info, 0, nullptr, &size));
  std::string str(size, 0);
  UR_CHECK_WEAK(urDeviceGetInfo(device, info, size, str.data(), nullptr));
  str.pop_back(); // std::string does not need a terminating NULL, remove it
                  // here
  std::cout << str << "\n";
}

inline void printDeviceUUID(ur_device_handle_t device, ur_device_info_t info) {
  std::cout << getDeviceInfoName(info) << ": ";
  size_t size;
  UR_CHECK_WEAK(urDeviceGetInfo(device, info, 0, nullptr, &size));
  std::vector<uint8_t> values(size / sizeof(uint8_t));
  UR_CHECK_WEAK(urDeviceGetInfo(device, info, size, values.data(), nullptr));
  for (size_t i = 0; i < values.size(); i++) {
    if (i == 4 || i == 6 || i == 8 || i == 10) {
      std::printf("-");
    }
    std::printf("%.2x", (uint32_t)values[i]);
  }
  std::printf("\n");
}
} // namespace urinfo
