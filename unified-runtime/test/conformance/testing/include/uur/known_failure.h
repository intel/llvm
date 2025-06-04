// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_CONFORMANCE_INCLUDE_KNOWN_FAILURE_H_INCLUDED
#define UR_CONFORMANCE_INCLUDE_KNOWN_FAILURE_H_INCLUDED

#include "uur/environment.h"
#include "uur/utils.h"
#include <string>
#include <string_view>
#include <vector>

namespace uur {
namespace detail {
struct AdapterInfo {
  uint32_t version;
  ur_backend_t backend;
};

inline AdapterInfo getAdapterInfo(ur_adapter_handle_t adapter) {
  AdapterInfo info;
  urAdapterGetInfo(adapter, UR_ADAPTER_INFO_VERSION, sizeof(info.version),
                   &info.version, nullptr);
  urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND, sizeof(info.backend),
                   &info.backend, nullptr);
  return info;
}
} // namespace detail

struct Matcher {
  Matcher(uint32_t adapterVersion, ur_backend_t backend,
          std::vector<std::string> checkNames)
      : adapterVersion(adapterVersion), backend(backend),
        names(std::move(checkNames)) {}

  bool matches(const detail::AdapterInfo &adapterInfo,
               const std::string &name) const {
    if (backend != adapterInfo.backend) {
      return false;
    }
    if (adapterVersion != adapterInfo.version) {
      return false;
    }
    if (names.empty()) {
      return true;
    }
    for (const auto &matcherName : names) {
      if (name.find(matcherName) != std::string::npos) {
        return true;
      }
    }
    return false;
  }

  uint32_t adapterVersion;
  ur_backend_t backend;
  std::vector<std::string> names;
};

struct OpenCL : Matcher {
  OpenCL(std::initializer_list<std::string> il)
      : Matcher(1, UR_BACKEND_OPENCL, {il.begin(), il.end()}) {}
};

struct LevelZero : Matcher {
  LevelZero(std::initializer_list<std::string> il)
      : Matcher(1, UR_BACKEND_LEVEL_ZERO, {il.begin(), il.end()}) {}
};

struct LevelZeroV2 : Matcher {
  LevelZeroV2(std::initializer_list<std::string> il)
      : Matcher(2, UR_BACKEND_LEVEL_ZERO, {il.begin(), il.end()}) {}
};

struct CUDA : Matcher {
  CUDA(std::initializer_list<std::string> il)
      : Matcher(1, UR_BACKEND_CUDA, {il.begin(), il.end()}) {}
};

struct HIP : Matcher {
  HIP(std::initializer_list<std::string> il)
      : Matcher(1, UR_BACKEND_HIP, {il.begin(), il.end()}) {}
};

struct NativeCPU : Matcher {
  NativeCPU(std::initializer_list<std::string> il)
      : Matcher(1, UR_BACKEND_NATIVE_CPU, {il.begin(), il.end()}) {}
};

inline bool isKnownFailureOn(ur_adapter_handle_t adapter,
                             const std::vector<Matcher> &matchers) {
  auto adapterInfo = detail::getAdapterInfo(adapter);
  for (const auto &matcher : matchers) {
    if (matcher.adapterVersion == adapterInfo.version &&
        matcher.backend == adapterInfo.backend) {
      return true;
    }
  }
  return false;
}

inline bool isKnownFailureOn(ur_platform_handle_t platform,
                             const std::vector<Matcher> &matchers) {
  ur_adapter_handle_t adapter = nullptr;
  urPlatformGetInfo(platform, UR_PLATFORM_INFO_ADAPTER,
                    sizeof(ur_adapter_handle_t), &adapter, nullptr);
  auto adapterInfo = detail::getAdapterInfo(adapter);
  std::string name;
  uur::GetPlatformInfo<std::string>(platform, UR_PLATFORM_INFO_NAME, name);
  for (const auto &matcher : matchers) {
    if (matcher.matches(adapterInfo, name)) {
      return true;
    }
  }
  return false;
}

template <class Param>
inline bool
isKnownFailureOn(const std::tuple<ur_platform_handle_t, Param> &param,
                 const std::vector<Matcher> &matchers) {
  ur_platform_handle_t platform = std::get<0>(param);
  return isKnownFailureOn(platform, matchers);
}

inline bool isKnownFailureOn(const DeviceTuple &param,
                             const std::vector<Matcher> &matchers) {
  auto adapterInfo = detail::getAdapterInfo(param.adapter);
  std::string name;
  uur::GetDeviceInfo<std::string>(param.device, UR_DEVICE_INFO_NAME, name);
  for (const auto &matcher : matchers) {
    if (matcher.matches(adapterInfo, name)) {
      return true;
    }
  }
  return false;
}

inline bool
isKnownFailureOn(std::tuple<ur_platform_handle_t, ur_device_handle_t> param,
                 const std::vector<Matcher> &matchers) {
  ur_adapter_handle_t adapter = nullptr;
  uur::GetPlatformInfo<ur_adapter_handle_t>(std::get<0>(param),
                                            UR_PLATFORM_INFO_ADAPTER, adapter);
  auto adapterInfo = detail::getAdapterInfo(adapter);
  std::string name;
  uur::GetDeviceInfo<std::string>(std::get<1>(param), UR_DEVICE_INFO_NAME,
                                  name);
  for (const auto &matcher : matchers) {
    if (matcher.matches(adapterInfo, name)) {
      return true;
    }
  }
  return false;
}

template <class Param>
inline bool isKnownFailureOn(const std::tuple<DeviceTuple, Param> &param,
                             const std::vector<Matcher> &matchers) {
  return isKnownFailureOn(std::get<0>(param), matchers);
}

inline std::string knownFailureMessage(ur_adapter_handle_t adapter) {
  std::string backend = uur::GetAdapterBackendName(adapter);
  return "Known failure on: " + backend;
}

inline std::string knownFailureMessage(ur_platform_handle_t platform) {
  ur_adapter_handle_t adapter = nullptr;
  urPlatformGetInfo(platform, UR_PLATFORM_INFO_ADAPTER,
                    sizeof(ur_adapter_handle_t), &adapter, nullptr);
  std::string backend = uur::GetAdapterBackendName(adapter);
  std::string platformName;
  uur::GetPlatformInfo<std::string>(platform, UR_PLATFORM_INFO_NAME,
                                    platformName);
  return "Known failure on: " + backend + ", " + platformName;
}

template <class Param>
inline std::string
knownFailureMessage(const std::tuple<ur_platform_handle_t, Param> &param) {
  ur_platform_handle_t platform = std::get<0>(param);
  return knownFailureMessage(platform);
}

inline std::string knownFailureMessage(const DeviceTuple &param) {
  std::string backend = uur::GetAdapterBackendName(param.adapter);
  std::string platformName;
  uur::GetPlatformInfo<std::string>(param.platform, UR_PLATFORM_INFO_NAME,
                                    platformName);
  std::string deviceName;
  uur::GetDeviceInfo<std::string>(param.device, UR_DEVICE_INFO_NAME,
                                  deviceName);
  return "Known failure on: " + backend + ", " + platformName + ", " +
         deviceName;
}

template <class Param>
inline std::string
knownFailureMessage(const std::tuple<DeviceTuple, Param> &param) {
  return knownFailureMessage(std::get<0>(param));
}

inline bool alsoRunKnownFailures() {
  if (const char *envvar = std::getenv("UR_CTS_ALSO_RUN_KNOWN_FAILURES")) {
    std::string_view value(envvar);
    return value == "1" || value == "ON" || value == "on" || value == "YES" ||
           value == "yes" || value == "true" || value == "TRUE";
  }
  return false;
}
} // namespace uur

#define UUR_KNOWN_FAILURE_ON_PARAM(param, ...)                                 \
  if (uur::isKnownFailureOn(param, {__VA_ARGS__})) {                           \
    auto message = uur::knownFailureMessage(param);                            \
    if (uur::alsoRunKnownFailures()) {                                         \
      std::cerr << message << "\n";                                            \
    } else {                                                                   \
      GTEST_SKIP() << message;                                                 \
    }                                                                          \
  }                                                                            \
  (void)0

#define UUR_KNOWN_FAILURE_ON(...)                                              \
  UUR_KNOWN_FAILURE_ON_PARAM(this->GetParam(), __VA_ARGS__)

#endif // UR_CONFORMANCE_INCLUDE_KNOWN_FAILURE_H_INCLUDED
