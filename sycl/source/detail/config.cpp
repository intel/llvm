//==---------------- config.cpp ---------------------------------*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/os_util.hpp>
#include <detail/config.hpp>

#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>

#define STRINGIFY_LINE_HELP(s) #s
#define STRINGIFY_LINE(s) STRINGIFY_LINE_HELP(s)

__SYCL_INLINE namespace cl {
namespace sycl {
namespace detail {

#ifndef SYCL_CONFIG_FILE_NAME
#define SYCL_CONFIG_FILE_NAME "sycl.conf"
#endif // SYCL_CONFIG_FILE_NAME

#define CONFIG(Name, MaxSize, CompileTimeDef)                                  \
  const char *SYCLConfigBase<Name>::MValueFromFile = nullptr;                  \
  char SYCLConfigBase<Name>::MStorage[MaxSize];                                \
  const char *const SYCLConfigBase<Name>::MCompileTimeDef =                    \
      getStrOrNullptr(STRINGIFY_LINE(CompileTimeDef));                         \
  const char *const SYCLConfigBase<Name>::MConfigName = STRINGIFY_LINE(Name);
#include "detail/config.def"
#undef CONFIG

#define MAX_CONFIG_NAME 256

static void initValue(const char *Key, const char *Value) {
#define CONFIG(Name, MaxSize, CompileTimeDef)                                  \
  if (0 == strncmp(Key, SYCLConfigBase<Name>::MConfigName, MAX_CONFIG_NAME)) { \
    strncpy(SYCLConfigBase<Name>::MStorage, Value, MaxSize);                   \
    SYCLConfigBase<Name>::MValueFromFile = SYCLConfigBase<Name>::MStorage;     \
    return;                                                                    \
  }
#include "detail/config.def"
#undef CONFIG
}

void readConfig() {
  static bool Initialized = false;
  if (Initialized)
    return;

  std::fstream File;
  if (const char *ConfigFile = getenv("SYCL_CONFIG_FILE_NAME"))
    File.open(ConfigFile, std::ios::in);
  else {
    const std::string LibSYCLDir = sycl::detail::OSUtil::getCurrentDSODir();
    File.open(LibSYCLDir + sycl::detail::OSUtil::DirSep + SYCL_CONFIG_FILE_NAME,
              std::ios::in);
  }

  if (File.is_open()) {
    // TODO: Use max size from macro instead of 256
    char Key[MAX_CONFIG_NAME] = {0}, Value[256] = {0};
    while (!File.eof()) {
      // Expected fromat:
      // ConfigName=Value\r
      // ConfigName=Value
      // TODO: Skip spaces before and after '='
      File.getline(Key, sizeof(Key), '=');
      if (File.fail()) {
        // Fail to process the line. Skip it completely and try next one.
        // Do we want to restore here? Or just throw an exception?
        File.clear(File.rdstate() & ~std::ios_base::failbit);
        File.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        continue;
      }
      File.getline(Value, sizeof(Value), '\n');

      if (File.fail()) {
        // Fail to process the value while config name is OK. It's likely that
        // value is too long. Currently just deal what we have got and ignore
        // remaining characters on the line.
        // Do we want to restore here? Or just throw an exception?
        File.clear(File.rdstate() & ~std::ios_base::failbit);
        File.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      }

      // Handle '\r' by nullifying it
      const std::streamsize ReadSybmols = File.gcount();
      if (ReadSybmols > 1 && '\r' == Value[ReadSybmols - 2])
        Value[ReadSybmols - 2] = '\0';

      initValue(Key, Value);
    }
  }
  Initialized = true;
}

// Prints configs name with their value
void dumpConfig() {
#define CONFIG(Name, MaxSize, CompileTimeDef)                                  \
  {                                                                            \
    const char *Val = SYCLConfig<Name>::get();                                 \
    std::cerr << SYCLConfigBase<Name>::MConfigName << " : "                    \
              << (Val ? Val : "unset") << std::endl;                           \
  }
#include "detail/config.def"
#undef CONFIG
}

} // namespace cl
} // namespace sycl
} // namespace detail

#undef STRINGIFY_LINE_HELP
#undef STRINGIFY_LINE
