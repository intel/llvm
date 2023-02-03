// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef UR_LEVEL_HPP
#define UR_LEVEL_HPP 1

#include <stdexcept>
#include <string>

namespace logger {

enum class Level { DEBUG,
                   INFO,
                   WARN,
                   ERR,
                   QUIET };

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
                                  {"error", Level::WARN}};

    for (auto const &item : lvl_names) {
        if (item.name.compare(name) == 0) {
            return item.lvl;
        }
    }
    throw std::invalid_argument(
        std::string("Parsing error: no valid log level for string '") + name +
        std::string("'.") +
        std::string(
            "\nValid log level names are: debug, info, warning and error"));
}

} // namespace logger

#endif /* UR_LEVEL_HPP */
