//==---------------- config.hpp - SYCL context ------------------*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/device_filter.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/info/info_desc.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string>
#include <utility>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

#ifdef DISABLE_CONFIG_FROM_ENV
constexpr bool ConfigFromEnvEnabled = false;
#else
constexpr bool ConfigFromEnvEnabled = true;
#endif // DISABLE_CONFIG_FROM_ENV

#ifdef DISABLE_CONFIG_FROM_CONFIG_FILE
constexpr bool ConfigFromFileEnabled = false;
#else
constexpr bool ConfigFromFileEnabled = true;
#endif // DISABLE_CONFIG_FROM_CONFIG_FILE

#ifdef DISABLE_CONFIG_FROM_COMPILE_TIME
constexpr bool ConfigFromCompileDefEnabled = false;
#else
constexpr bool ConfigFromCompileDefEnabled = true;
#endif // DISABLE_CONFIG_FROM_COMPILE_TIME

// Enum of config IDs for accessing other arrays
enum ConfigID {
  START = 0,
#define CONFIG(name, ...) name,
#include "config.def"
#undef CONFIG
  END
};

// Consider strings starting with __ as unset
constexpr const char *getStrOrNullptr(const char *Str) {
  return (Str[0] == '_' && Str[1] == '_') ? nullptr : Str;
}

// Intializes configs from the configuration file
void readConfig();

template <ConfigID Config> class SYCLConfigBase;

#define CONFIG(Name, MaxSize, CompileTimeDef)                                  \
  template <> class SYCLConfigBase<Name> {                                     \
  public:                                                                      \
    /*Preallocated storage for config value which is extracted from a config   \
     * file*/                                                                  \
    static char MStorage[MaxSize + 1];                                         \
    /*Points to the storage if config is set in the file, nullptr otherwise*/  \
    static const char *MValueFromFile;                                         \
    /*The name of the config*/                                                 \
    static const char *const MConfigName;                                      \
    /*Points to the value which is set during compilation, nullptr otherwise.  \
     * Detection of whether a value is set or not is based on checking the     \
     * beginning of the string, if it starts with double underscore(__) the    \
     * value is not set.*/                                                     \
    static const char *const MCompileTimeDef;                                  \
                                                                               \
    static const char *getRawValue() {                                         \
      if (ConfigFromEnvEnabled)                                                \
        if (const char *ValStr = getenv(MConfigName))                          \
          return ValStr;                                                       \
                                                                               \
      if (ConfigFromFileEnabled) {                                             \
        readConfig();                                                          \
        if (MValueFromFile)                                                    \
          return MValueFromFile;                                               \
      }                                                                        \
                                                                               \
      if (ConfigFromCompileDefEnabled && MCompileTimeDef)                      \
        return MCompileTimeDef;                                                \
                                                                               \
      return nullptr;                                                          \
    }                                                                          \
  };
#include "config.def"
#undef CONFIG

template <ConfigID Config> class SYCLConfig {
  using BaseT = SYCLConfigBase<Config>;

public:
  static const char *get() {
    static const char *ValStr = BaseT::getRawValue();
    return ValStr;
  }
};

template <> class SYCLConfig<SYCL_BE> {
  using BaseT = SYCLConfigBase<SYCL_BE>;

public:
  static backend *get() {
    static bool Initialized = false;
    static backend *BackendPtr = nullptr;

    // Configuration parameters are processed only once, like reading a string
    // from environment and converting it into a typed object.
    if (Initialized)
      return BackendPtr;

    const char *ValStr = BaseT::getRawValue();
    const std::array<std::pair<std::string, backend>, 4> SyclBeMap = {
        {{"PI_OPENCL", backend::opencl},
         {"PI_LEVEL_ZERO", backend::level_zero},
         {"PI_LEVEL0", backend::level_zero}, // for backward compatibility
         {"PI_CUDA", backend::cuda}}};
    if (ValStr) {
      auto It = std::find_if(
          std::begin(SyclBeMap), std::end(SyclBeMap),
          [&ValStr](const std::pair<std::string, backend> &element) {
            return element.first == ValStr;
          });
      if (It == SyclBeMap.end())
        pi::die("Invalid backend. "
                "Valid values are PI_OPENCL/PI_LEVEL_ZERO/PI_CUDA");
      static backend Backend = It->second;
      BackendPtr = &Backend;
    }
    Initialized = true;
    return BackendPtr;
  }
};

template <> class SYCLConfig<SYCL_PI_TRACE> {
  using BaseT = SYCLConfigBase<SYCL_PI_TRACE>;

public:
  static int get() {
    static bool Initialized = false;
    // We don't use TraceLevel enum here because user can provide any bitmask
    // which can correspond to several enum values.
    static int Level = 0; // No tracing by default

    // Configuration parameters are processed only once, like reading a string
    // from environment and converting it into a typed object.
    if (Initialized)
      return Level;

    const char *ValStr = BaseT::getRawValue();
    Level = (ValStr ? std::atoi(ValStr) : 0);
    Initialized = true;
    return Level;
  }
};

template <> class SYCLConfig<SYCL_DEVICE_FILTER> {
  using BaseT = SYCLConfigBase<SYCL_DEVICE_FILTER>;

public:
  static device_filter_list *get() {
    static bool Initialized = false;
    static device_filter_list *FilterList = nullptr;

    // Configuration parameters are processed only once, like reading a string
    // from environment and converting it into a typed object.
    if (Initialized) {
      return FilterList;
    }

    const char *ValStr = BaseT::getRawValue();
    if (ValStr) {
      static device_filter_list DFL{ValStr};
      FilterList = &DFL;
    }

    // TODO: remove the following code when we remove the support for legacy
    // env vars.
    // Emit the deprecation warning message if SYCL_BE or SYCL_DEVICE_TYPE is
    // set.
    if (SYCLConfig<SYCL_BE>::get() || getenv("SYCL_DEVICE_TYPE")) {
      std::cerr << "\nWARNING: The legacy environment variables SYCL_BE and "
                   "SYCL_DEVICE_TYPE are deprecated. Please use "
                   "SYCL_DEVICE_FILTER instead. For details, please refer to "
                   "https://github.com/intel/llvm/blob/sycl/sycl/doc/"
                   "EnvironmentVariables.md\n\n";
    }

    // As mentioned above, configuration parameters are processed only once.
    // If multiple threads are checking this env var at the same time,
    // they will end up setting the configration to the same value.
    // If other threads check after one thread already set configration,
    // the threads will get the same value as the first thread.
    Initialized = true;
    return FilterList;
  }
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
