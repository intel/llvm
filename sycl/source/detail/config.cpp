//==---------------- config.cpp ---------------------------------*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <detail/config.hpp>

#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

#ifndef SYCL_CONFIG_FILE_NAME
#define SYCL_CONFIG_FILE_NAME "sycl.conf"
#endif // SYCL_CONFIG_FILE_NAME

#define CONFIG(Name, MaxSize, CompileTimeDef)                                  \
  const char *SYCLConfigBase<Name>::MValueFromFile = nullptr;                  \
  char SYCLConfigBase<Name>::MStorage[MaxSize + 1];                            \
  const char *const SYCLConfigBase<Name>::MCompileTimeDef =                    \
      getStrOrNullptr(__SYCL_STRINGIFY_LINE(CompileTimeDef));                  \
  const char *const SYCLConfigBase<Name>::MConfigName =                        \
      __SYCL_STRINGIFY_LINE(Name);
#include "detail/config.def"
#undef CONFIG

constexpr int MAX_CONFIG_NAME = 256;
constexpr int MAX_CONFIG_VALUE = 256;

static void initValue(const char *Key, const char *Value) {
#define CONFIG(Name, MaxSize, CompileTimeDef)                                  \
  if (0 == strncmp(Key, SYCLConfigBase<Name>::MConfigName, MAX_CONFIG_NAME)) { \
    strncpy(SYCLConfigBase<Name>::MStorage, Value, MaxSize);                   \
    SYCLConfigBase<Name>::MStorage[MaxSize] = '\0';                            \
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
    char Key[MAX_CONFIG_NAME] = {0}, Value[MAX_CONFIG_VALUE] = {0};
    std::string BufString;
    std::size_t Position = std::string::npos;
    while (!File.eof()) {
      // Expected format:
      // ConfigName=Value\r
      // ConfigName=Value
      // TODO: Skip spaces before and after '='
      std::getline(File, BufString);
      if  (File.fail() && !File.eof()) {
        // Fail to process the line.
        File.clear(File.rdstate() & ~std::ios_base::failbit);
        File.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        throw sycl::exception(
            make_error_code(errc::runtime),
            "An error occurred while attempting to read a line");
      }
      // Skip lines with a length = 0 | which have a comment | don't have "="
      if ((BufString.length() == 0) ||
          (BufString.find("#") != std::string::npos) ||
          (BufString.find("=") == std::string::npos)) {
        continue;
      }
      // Finding the position of '='
      Position = BufString.find("=");
      // Checking that the variable name is less than MAX_CONFIG_NAME and more
      // than zero character
      if ((Position <= MAX_CONFIG_NAME) && (Position > 0)) {
        // Checking for spaces at the beginning and end of the line,
        // before and after '='
        if ((BufString[0] == ' ') ||
            (BufString[BufString.length() - 1] == ' ') ||
            (BufString[Position - 1] == ' ') ||
            (BufString[Position + 1] == ' ')) {
          throw sycl::exception(make_error_code(errc::runtime),
                                "SPACE found at the beginning/end of the line "
                                "or before/after '='");
        }
        // Checking that the value is less than MAX_CONFIG_VALUE and
        // more than zero character
        if ((BufString.length() - (Position + 1) <= MAX_CONFIG_VALUE) &&
            (BufString.length() != Position + 1)) {
          // Creating pairs of (key, value)
          BufString.copy(Key, Position, 0);
          Key[Position] = '\0';
          BufString.copy(Value, BufString.length() - (Position + 1),
                         Position + 1);
          Value[BufString.length() - (Position + 1)] = '\0';
        } else {
          throw sycl::exception(
              make_error_code(errc::runtime),
              "The value contains more than " +
                  std::to_string(MAX_CONFIG_VALUE) +
                  " characters or does not contain them at all");
        }
      } else {
        throw sycl::exception(make_error_code(errc::runtime),
                              "Variable name is more than " +
                                  std::to_string(MAX_CONFIG_NAME) +
                                  " or less than one character");
      }
      // Handle '\r' by nullifying it
      if ('\r' == Value[BufString.length() - Position - 3])
        Value[BufString.length() - Position - 3] = '\0';

      initValue(Key, Value);
    }
  }
  Initialized = true;
}

// Prints configs name with their value
void dumpConfig() {
#define CONFIG(Name, MaxSize, CompileTimeDef)                                  \
  {                                                                            \
    const char *Val = SYCLConfigBase<Name>::getRawValue();                     \
    std::cerr << SYCLConfigBase<Name>::MConfigName << " : "                    \
              << (Val ? Val : "unset") << std::endl;                           \
  }
#include "detail/config.def"
#undef CONFIG
}

} // __SYCL_INLINE_NAMESPACE(cl)
} // namespace sycl
} // namespace detail
