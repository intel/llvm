// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef UR_LEVEL_HPP
#define UR_LEVEL_HPP 1

#include <stdexcept>
#include <string>

namespace logger {

enum class Level { DEBUG, INFO, WARN, ERR, QUIET };

inline constexpr auto level_to_str(Level level) {
  switch (level) {
  case Level::DEBUG:
    return "DEBUG";
  case Level::INFO:
    return "INFO";
  case Level::WARN:
    return "WARNING";
  case Level::ERR:
    return "ERROR";
  case Level::QUIET:
    return "QUIET";
  default:
    return "";
  }
}

inline auto str_to_level(std::string name) {
  struct lvl_name {
    std::string name;
    Level lvl;
  };

  const lvl_name lvl_names[] = {{"debug", Level::DEBUG},
                                {"info", Level::INFO},
                                {"warning", Level::WARN},
                                {"error", Level::ERR},
                                {"quiet", Level::QUIET}};

  for (auto const &item : lvl_names) {
    if (item.name.compare(name) == 0) {
      return item.lvl;
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
