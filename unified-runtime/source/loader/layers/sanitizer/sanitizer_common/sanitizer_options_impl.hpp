/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer_options_impl.hpp
 *
 */

#pragma once

#include "logger/ur_logger.hpp"
#include "ur/ur.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ur_sanitizer_layer {

namespace options {

struct OptionParser {
  logger::Logger &Logger;
  const EnvVarMap &EnvMap;

  OptionParser(const EnvVarMap &EnvMap, logger::Logger &Logger)
      : Logger(Logger), EnvMap(EnvMap) {}

  const char *TrueStrings[2] = {"1", "true"};
  const char *FalseStrings[2] = {"0", "false"};

  void InplaceToLower(std::string &S) {
    std::transform(S.begin(), S.end(), S.begin(),
                   [](unsigned char C) { return std::tolower(C); });
  }

  bool IsTrue(const std::string &S) {
    return std::any_of(std::begin(TrueStrings), std::end(TrueStrings),
                       [&](const char *CS) { return S == CS; });
  }

  bool IsFalse(const std::string &S) {
    return std::any_of(std::begin(FalseStrings), std::end(FalseStrings),
                       [&](const char *CS) { return S == CS; });
  }
  void ParseBool(const std::string &Name, bool &Result) {
    auto KV = EnvMap.find(Name);
    if (KV != EnvMap.end()) {
      auto ValueStr = KV->second.front();
      InplaceToLower(ValueStr);
      if (IsTrue(ValueStr)) {
        Result = true;
      } else if (IsFalse(ValueStr)) {
        Result = false;
      } else {
        std::stringstream SS;
        SS << "\"" << Name << "\" is set to \"" << ValueStr
           << "\", which is not an valid setting. ";
        SS << "Acceptable input are: for enable, use:";
        for (auto &S : TrueStrings) {
          SS << " \"" << S << "\"";
        }
        SS << "; ";
        SS << "for disable, use:";
        for (auto &S : FalseStrings) {
          SS << " \"" << S << "\"";
        }
        SS << ".";
        UR_LOG_L(Logger, ERR, SS.str().c_str());
        die("Sanitizer failed to parse options.\n");
      }
    }
  }

  void ParseUint64(const std::string &Name, uint64_t &Result, uint64_t Min = 0,
                   uint64_t Max = UINT64_MAX) {
    auto KV = EnvMap.find(Name);
    if (KV != EnvMap.end()) {
      const auto &ValueStr = KV->second.front();
      try {
        // Check for possible negative numbers(stoul would not throw for
        // negative number)
        if (ValueStr[0] == '-') {
          throw std::out_of_range("Negative number");
        }

        uint64_t Value = std::stoul(ValueStr.c_str());

        if (Value < Min) {
          UR_LOG_L(Logger, WARN,
                   "The valid range of \"{}\" is [{}, {}]. "
                   "Setting to the minimum value {}.",
                   Name, Min, Max, Min);
          Result = Min;
        } else if (Value > Max) {
          UR_LOG_L(Logger, WARN,
                   "The valid range of \"{}\" is [{}, {}]. "
                   "Setting to the maximum value {}.",
                   Name, Min, Max, Max);
          Result = Max;
        } else {
          Result = Value;
        }
      } catch (...) {
        UR_LOG_L(Logger, ERR,
                 "The valid range of \"{}\" is [{}, {}]. Failed "
                 "to parse the value \"{}\".",
                 Name, Min, Max, ValueStr);
        die("Sanitizer failed to parse options.\n");
      }
    }
  }
};
} // namespace options

} // namespace ur_sanitizer_layer
