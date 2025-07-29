// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef UR_LEVEL_HPP
#define UR_LEVEL_HPP 1

#include <stdexcept>
#include <string>
#include <ur_api.h>

namespace logger {

inline constexpr auto level_to_str(ur_logger_level_t level) {
  switch (level) {
  case UR_LOGGER_LEVEL_DEBUG:
    return "DEBUG";
  case UR_LOGGER_LEVEL_INFO:
    return "INFO";
  case UR_LOGGER_LEVEL_WARN:
    return "WARNING";
  case UR_LOGGER_LEVEL_ERROR:
    return "ERROR";
  case UR_LOGGER_LEVEL_QUIET:
    return "QUIET";
  default:
    return "";
  }
}

inline auto str_to_level(std::string name) {
  struct level_name {
    std::string name;
    ur_logger_level_t level;
  };

  const level_name level_names[] = {{"debug", UR_LOGGER_LEVEL_DEBUG},
                                    {"info", UR_LOGGER_LEVEL_INFO},
                                    {"warning", UR_LOGGER_LEVEL_WARN},
                                    {"error", UR_LOGGER_LEVEL_ERROR},
                                    {"quiet", UR_LOGGER_LEVEL_QUIET}};

  for (auto const &item : level_names) {
    if (item.name.compare(name) == 0) {
      return item.level;
    }
  }
  throw std::invalid_argument(
      std::string("Parsing error: no valid log level for string '") + name +
      std::string("'.") +
      std::string("\nValid log level names are: debug, info, warning, error, "
                  "and quiet"));
}

} // namespace logger

#endif /* UR_LEVEL_HPP */
