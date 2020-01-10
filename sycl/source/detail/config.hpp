//==---------------- config.hpp - SYCL context ------------------*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdlib>

__SYCL_INLINE namespace cl {
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

template <ConfigID Config> class SYCLConfigBase;

#define CONFIG(Name, MaxSize, CompileTimeDef)                                  \
  template <> class SYCLConfigBase<Name> {                                     \
  public:                                                                      \
    /*Preallocated storage for config value which is extracted from a config   \
     * file*/                                                                  \
    static char MStorage[MaxSize];                                             \
    /*Points to the storage if config is set in the file, nullptr otherwise*/  \
    static const char *MValueFromFile;                                         \
    /*The name of the config*/                                                 \
    static const char *const MConfigName;                                      \
    /*Points to the value which is set during compilation, nullptr otherwise.  \
     * Detection of whether a value is set or not is based on checking the     \
     * beginning of the string, if it starts with double underscore(__) the    \
     * value is not set.*/                                                     \
    static const char *const MCompileTimeDef;                                  \
  };
#include "config.def"
#undef CONFIG

// Intializes configs from the configuration file
void readConfig();

template <ConfigID Config> class SYCLConfig {
  using BaseT = SYCLConfigBase<Config>;

public:
  static const char *get() {
    const char *ValStr = getRawValue();
    return ValStr;
  }

private:
  static const char *getRawValue() {
    if (ConfigFromEnvEnabled)
      if (const char *ValStr = getenv(BaseT::MConfigName))
        return ValStr;

    if (ConfigFromFileEnabled) {
      readConfig();
      if (BaseT::MValueFromFile)
        return BaseT::MValueFromFile;
    }

    if (ConfigFromCompileDefEnabled && BaseT::MCompileTimeDef)
      return BaseT::MCompileTimeDef;

    return nullptr;
  }
};

} // namespace cl
} // namespace sycl
} // namespace detail
