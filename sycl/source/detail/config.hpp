//==---------------- config.hpp - SYCL config -------------------*- C++-*---==//
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
#include <detail/global_handler.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <mutex>
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

constexpr int MAX_CONFIG_NAME = 256;
constexpr int MAX_CONFIG_VALUE = 256;

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
void readConfig(bool ForceInitialization = false);

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
  static const char *get() { return getCachedValue(); }

  static void reset() { (void)getCachedValue(/*ResetCache=*/true); }

  static const char *getName() { return BaseT::MConfigName; }

private:
  static const char *getCachedValue(bool ResetCache = false) {
    static const char *ValStr = BaseT::getRawValue();
    if (ResetCache)
      ValStr = BaseT::getRawValue();
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
    const std::array<std::pair<std::string, backend>, 6> SyclBeMap = {
        {{"PI_OPENCL", backend::opencl},
         {"PI_LEVEL_ZERO", backend::ext_oneapi_level_zero},
         {"PI_LEVEL0", backend::ext_oneapi_level_zero}, // for backward
                                                        // compatibility
         {"PI_CUDA", backend::ext_oneapi_cuda},
         {"PI_ESIMD_EMULATOR", backend::ext_intel_esimd_emulator},
         {"PI_HIP", backend::ext_oneapi_hip}}};
    if (ValStr) {
      auto It = std::find_if(
          std::begin(SyclBeMap), std::end(SyclBeMap),
          [&ValStr](const std::pair<std::string, backend> &element) {
            return element.first == ValStr;
          });
      if (It == SyclBeMap.end())
        pi::die("Invalid backend. "
                "Valid values are "
                "PI_OPENCL/PI_LEVEL_ZERO/PI_CUDA/PI_ESIMD_EMULATOR/PI_HIP");
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

template <> class SYCLConfig<SYCL_RT_WARNING_LEVEL> {
  using BaseT = SYCLConfigBase<SYCL_RT_WARNING_LEVEL>;

public:
  static unsigned int get() { return getCachedValue(); }

  static void reset() { (void)getCachedValue(true); }

private:
  static unsigned int getCachedValue(bool ResetCache = false) {
    const auto Parser = []() {
      const char *ValStr = BaseT::getRawValue();
      int SignedLevel = ValStr ? std::atoi(ValStr) : 0;
      return SignedLevel >= 0 ? SignedLevel : 0;
    };

    static unsigned int Level = Parser();
    if (ResetCache)
      Level = Parser();

    return Level;
  }
};

template <> class SYCLConfig<SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE> {
  using BaseT = SYCLConfigBase<SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE>;

public:
  static bool get() {
    static const char *ValStr = BaseT::getRawValue();
    return ValStr != nullptr;
  }
};

template <> class SYCLConfig<SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING> {
  using BaseT = SYCLConfigBase<SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING>;

public:
  static bool get() {
    static const char *ValStr = BaseT::getRawValue();
    return ValStr != nullptr;
  }
};

template <> class SYCLConfig<SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS> {
  using BaseT = SYCLConfigBase<SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS>;

private:
public:
  static void GetSettings(size_t &MinFactor, size_t &GoodFactor,
                          size_t &MinRange) {
    static const char *RoundParams = BaseT::getRawValue();
    if (RoundParams == nullptr)
      return;

    static bool ProcessedFactors = false;
    static size_t MF;
    static size_t GF;
    static size_t MR;
    if (!ProcessedFactors) {
      // Parse optional parameters of this form (all values required):
      // MinRound:PreferredRound:MinRange
      std::string Params(RoundParams);
      size_t Pos = Params.find(':');
      if (Pos != std::string::npos) {
        MF = std::stoi(Params.substr(0, Pos));
        Params.erase(0, Pos + 1);
        Pos = Params.find(':');
        if (Pos != std::string::npos) {
          GF = std::stoi(Params.substr(0, Pos));
          Params.erase(0, Pos + 1);
          MR = std::stoi(Params);
        }
      }
      ProcessedFactors = true;
    }
    MinFactor = MF;
    GoodFactor = GF;
    MinRange = MR;
  }
};

// Array is used by SYCL_DEVICE_FILTER and SYCL_DEVICE_ALLOWLIST
const std::array<std::pair<std::string, info::device_type>, 5> &
getSyclDeviceTypeMap();

// Array is used by SYCL_DEVICE_FILTER and SYCL_DEVICE_ALLOWLIST
const std::array<std::pair<std::string, backend>, 7> &getSyclBeMap();

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
      FilterList = &GlobalHandler::instance().getDeviceFilterList(ValStr);
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

template <> class SYCLConfig<SYCL_ENABLE_DEFAULT_CONTEXTS> {
  using BaseT = SYCLConfigBase<SYCL_ENABLE_DEFAULT_CONTEXTS>;

public:
  static bool get() {
    constexpr bool DefaultValue = true;
    const char *ValStr = getCachedValue();

    if (!ValStr)
      return DefaultValue;

    return ValStr[0] == '1';
  }

  static void reset() { (void)getCachedValue(/*ResetCache=*/true); }

  static const char *getName() { return BaseT::MConfigName; }

private:
  static const char *getCachedValue(bool ResetCache = false) {
    static const char *ValStr = BaseT::getRawValue();
    if (ResetCache)
      ValStr = BaseT::getRawValue();
    return ValStr;
  }
};

template <> class SYCLConfig<SYCL_QUEUE_THREAD_POOL_SIZE> {
  using BaseT = SYCLConfigBase<SYCL_QUEUE_THREAD_POOL_SIZE>;

public:
  static int get() {
    static int Value = [] {
      const char *ValueStr = BaseT::getRawValue();

      int Result = 1;

      if (ValueStr)
        try {
          Result = std::stoi(ValueStr);
        } catch (...) {
          throw invalid_parameter_error(
              "Invalid value for SYCL_QUEUE_THREAD_POOL_SIZE environment "
              "variable: value should be a number",
              PI_INVALID_VALUE);
        }

      if (Result < 1)
        throw invalid_parameter_error(
            "Invalid value for SYCL_QUEUE_THREAD_POOL_SIZE environment "
            "variable: value should be larger than zero",
            PI_INVALID_VALUE);

      return Result;
    }();

    return Value;
  }
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
