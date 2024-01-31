//==---------------- config.hpp - SYCL config -------------------*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/global_handler.hpp>
#include <sycl/backend_types.hpp>
#include <sycl/detail/defines.hpp>
#include <sycl/detail/device_filter.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/exception.hpp>
#include <sycl/info/info_desc.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>

namespace sycl {
inline namespace _V1 {
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
constexpr int MAX_CONFIG_VALUE = 1024;

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

#define INVALID_CONFIG_EXCEPTION(BASE, MSG)                                    \
  sycl::exception(sycl::make_error_code(sycl::errc::invalid),                  \
                  "Invalid value for " + std::string{BASE::MConfigName} +      \
                      " environment variable: " + MSG)

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

// Array is used by SYCL_DEVICE_FILTER and SYCL_DEVICE_ALLOWLIST and
// ONEAPI_DEVICE_SELECTOR
const std::array<std::pair<std::string, info::device_type>, 6> &
getSyclDeviceTypeMap();

// Array is used by SYCL_DEVICE_FILTER and SYCL_DEVICE_ALLOWLIST and
// ONEAPI_DEVICE_SELECTOR
const std::array<std::pair<std::string, backend>, 8> &getSyclBeMap();

// ---------------------------------------
// ONEAPI_DEVICE_SELECTOR support
template <> class SYCLConfig<ONEAPI_DEVICE_SELECTOR> {
  using BaseT = SYCLConfigBase<ONEAPI_DEVICE_SELECTOR>;

public:
  static ods_target_list *get() {
    // Configuration parameters are processed only once, like reading a string
    // from environment and converting it into a typed object.
    static bool Initialized = false;
    static ods_target_list *DeviceTargets = nullptr;

    if (Initialized) {
      return DeviceTargets;
    }
    const char *ValStr = BaseT::getRawValue();
    if (ValStr) {
      DeviceTargets =
          &GlobalHandler::instance().getOneapiDeviceSelectorTargets(ValStr);
    }
    Initialized = true;
    return DeviceTargets;
  }
};

// ---------------------------------------
// SYCL_DEVICE_FILTER support

template <>
class __SYCL2020_DEPRECATED("Use SYCLConfig<ONEAPI_DEVICE_SELECTOR> instead")
    SYCLConfig<SYCL_DEVICE_FILTER> {
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

      std::cerr
          << "\nWARNING: The enviroment variable SYCL_DEVICE_FILTER"
             " is deprecated. Please use ONEAPI_DEVICE_SELECTOR instead.\n"
             "For more details, please refer to:\n"
             "https://github.com/intel/llvm/blob/sycl/sycl/doc/"
             "EnvironmentVariables.md#oneapi_device_selector\n\n";

      FilterList = &GlobalHandler::instance().getDeviceFilterList(ValStr);
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

  static void resetWithValue(const char *Val) {
    (void)getCachedValue(/*ResetCache=*/true, Val);
  }

  static const char *getName() { return BaseT::MConfigName; }

private:
  static const char *getCachedValue(bool ResetCache = false,
                                    const char *Val = nullptr) {
    static const char *ValStr = BaseT::getRawValue();
    if (ResetCache) {
      ValStr = (Val != nullptr) ? Val : BaseT::getRawValue();
    }
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
              PI_ERROR_INVALID_VALUE);
        }

      if (Result < 1)
        throw invalid_parameter_error(
            "Invalid value for SYCL_QUEUE_THREAD_POOL_SIZE environment "
            "variable: value should be larger than zero",
            PI_ERROR_INVALID_VALUE);

      return Result;
    }();

    return Value;
  }
};

template <> class SYCLConfig<SYCL_CACHE_PERSISTENT> {
  using BaseT = SYCLConfigBase<SYCL_CACHE_PERSISTENT>;

public:
  static constexpr bool Default = false; // default is disabled

  static bool get() { return getCachedValue(); }

  static void reset() { (void)getCachedValue(/*ResetCache=*/true); }

  static const char *getName() { return BaseT::MConfigName; }

private:
  static bool parseValue() {
    // Check if deprecated opt-out env var is used, then warn.
    if (SYCLConfig<SYCL_CACHE_DISABLE_PERSISTENT>::get()) {
      std::cerr
          << "WARNING: " << SYCLConfig<SYCL_CACHE_DISABLE_PERSISTENT>::getName()
          << " environment variable is deprecated "
          << "and has no effect. By default, persistent device code caching is "
          << (Default ? "enabled." : "disabled.") << " Use " << getName()
          << "=1/0 to enable/disable.\n";
    }

    const char *ValStr = BaseT::getRawValue();
    if (!ValStr)
      return Default;
    if (strlen(ValStr) != 1 || (ValStr[0] != '0' && ValStr[0] != '1')) {
      std::string Msg =
          std::string{"Invalid value for bool configuration variable "} +
          getName() + std::string{": "} + ValStr;
      throw runtime_error(Msg, PI_ERROR_INVALID_OPERATION);
    }
    return ValStr[0] == '1';
  }

  static bool getCachedValue(bool ResetCache = false) {
    static bool Val = parseValue();
    if (ResetCache)
      Val = parseValue();
    return Val;
  }
};

template <> class SYCLConfig<SYCL_CACHE_DIR> {
  using BaseT = SYCLConfigBase<SYCL_CACHE_DIR>;

public:
  static std::string get() { return getCachedValue(); }

  static void reset() { (void)getCachedValue(/*ResetCache=*/true); }

  static const char *getName() { return BaseT::MConfigName; }

private:
  // If environment variables are not available return an empty string to
  // identify that cache is not available.
  static std::string parseValue() {
    const char *RootDir = BaseT::getRawValue();
    if (RootDir)
      return RootDir;

    constexpr char DeviceCodeCacheDir[] = "/libsycl_cache";

#if defined(__SYCL_RT_OS_LINUX)
    const char *CacheDir = std::getenv("XDG_CACHE_HOME");
    const char *HomeDir = std::getenv("HOME");
    if (!CacheDir && !HomeDir)
      return {};
    std::string Res{
        std::string(CacheDir ? CacheDir : (std::string(HomeDir) + "/.cache")) +
        DeviceCodeCacheDir};
#else
    const char *AppDataDir = std::getenv("AppData");
    if (!AppDataDir)
      return {};
    std::string Res{std::string(AppDataDir) + DeviceCodeCacheDir};
#endif
    return Res;
  }

  static std::string getCachedValue(bool ResetCache = false) {
    static std::string Val = parseValue();
    if (ResetCache)
      Val = parseValue();
    return Val;
  }
};

template <> class SYCLConfig<SYCL_REDUCTION_PREFERRED_WORKGROUP_SIZE> {
  using BaseT = SYCLConfigBase<SYCL_REDUCTION_PREFERRED_WORKGROUP_SIZE>;

  struct ParsedValue {
    size_t CPU = 0;
    size_t GPU = 0;
    size_t Accelerator = 0;
  };

public:
  static size_t get(info::device_type DeviceType) {
    ParsedValue Value = getCachedValue();
    return getRefByDeviceType(Value, DeviceType);
  }

  static void reset() { (void)getCachedValue(/*ResetCache=*/true); }

  static const char *getName() { return BaseT::MConfigName; }

private:
  static size_t &getRefByDeviceType(ParsedValue &Value,
                                    info::device_type DeviceType) {
    switch (DeviceType) {
    case info::device_type::cpu:
      return Value.CPU;
    case info::device_type::gpu:
      return Value.GPU;
    case info::device_type::accelerator:
      return Value.Accelerator;
    default:
      // Expect to get here if user used wrong device type. Include wildcard
      // in the message even though it's handled in the caller.
      throw INVALID_CONFIG_EXCEPTION(
          BaseT, "Device types must be \"cpu\", \"gpu\", \"acc\", or \"*\".");
    }
  }

  static ParsedValue parseValue() {
    const char *ValueRaw = BaseT::getRawValue();
    ParsedValue Result{};

    // Default to 0 to signify an unset value.
    if (!ValueRaw)
      return Result;

    std::string ValueStr{ValueRaw};
    auto DeviceTypeMap = getSyclDeviceTypeMap();

    // Iterate over all configurations.
    size_t Start = 0, End = 0;
    do {
      End = ValueStr.find(',', Start);
      if (End == std::string::npos)
        End = ValueStr.size();

      // Get a substring of the current configuration pair.
      std::string DeviceConfigStr = ValueStr.substr(Start, End - Start);

      // Find the delimiter in the configuration pair.
      size_t ConfigDelimLoc = DeviceConfigStr.find(':');
      if (ConfigDelimLoc == std::string::npos)
        throw INVALID_CONFIG_EXCEPTION(
            BaseT, "Device-value pair \"" + DeviceConfigStr +
                       "\" does not contain the ':' delimiter.");

      // Split configuration pair into its constituents.
      std::string DeviceConfigTypeStr =
          DeviceConfigStr.substr(0, ConfigDelimLoc);
      std::string DeviceConfigValueStr = DeviceConfigStr.substr(
          ConfigDelimLoc + 1, DeviceConfigStr.size() - ConfigDelimLoc - 1);

      // Find the device type in the "device type map".
      auto DeviceTypeIter = std::find_if(
          std::begin(DeviceTypeMap), std::end(DeviceTypeMap),
          [&](auto Element) { return DeviceConfigTypeStr == Element.first; });
      if (DeviceTypeIter == DeviceTypeMap.end())
        throw INVALID_CONFIG_EXCEPTION(
            BaseT,
            "\"" + DeviceConfigTypeStr + "\" is not a recognized device type.");

      // Parse the configuration value.
      int DeviceConfigValue = 1;
      try {
        DeviceConfigValue = std::stoi(DeviceConfigValueStr);
      } catch (...) {
        throw INVALID_CONFIG_EXCEPTION(
            BaseT, "Value \"" + DeviceConfigValueStr + "\" must be a number");
      }

      if (DeviceConfigValue < 1)
        throw INVALID_CONFIG_EXCEPTION(BaseT,
                                       "Value \"" + DeviceConfigValueStr +
                                           "\" must be larger than zero");

      if (DeviceTypeIter->second == info::device_type::all) {
        // Set all configuration values if we got the device-type wildcard.
        Result.GPU = DeviceConfigValue;
        Result.CPU = DeviceConfigValue;
        Result.Accelerator = DeviceConfigValue;
      } else {
        // Try setting the corresponding configuration.
        getRefByDeviceType(Result, DeviceTypeIter->second) = DeviceConfigValue;
      }

      // Move to the start of the next configuration. If the start is outside
      // the full value string we are done.
      Start = End + 1;
    } while (Start < ValueStr.size());
    return Result;
  }

  static ParsedValue getCachedValue(bool ResetCache = false) {
    static ParsedValue Val = parseValue();
    if (ResetCache)
      Val = parseValue();
    return Val;
  }
};

template <> class SYCLConfig<SYCL_ENABLE_FUSION_CACHING> {
  using BaseT = SYCLConfigBase<SYCL_ENABLE_FUSION_CACHING>;

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

template <> class SYCLConfig<SYCL_CACHE_IN_MEM> {
  using BaseT = SYCLConfigBase<SYCL_CACHE_IN_MEM>;

public:
  static constexpr bool Default = true; // default is true
  static bool get() { return getCachedValue(); }
  static const char *getName() { return BaseT::MConfigName; }

private:
  static bool parseValue() {
    const char *ValStr = BaseT::getRawValue();
    if (!ValStr)
      return Default;
    if (strlen(ValStr) != 1 || (ValStr[0] != '0' && ValStr[0] != '1')) {
      std::string Msg =
          std::string{"Invalid value for bool configuration variable "} +
          getName() + std::string{": "} + ValStr;
      throw runtime_error(Msg, PI_ERROR_INVALID_OPERATION);
    }
    return ValStr[0] == '1';
  }

  static bool getCachedValue() {
    static bool Val = parseValue();
    return Val;
  }
};

#undef INVALID_CONFIG_EXCEPTION

} // namespace detail
} // namespace _V1
} // namespace sycl
