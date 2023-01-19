// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef UR_LEVEL_HPP
#define UR_LEVEL_HPP 1

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

} // namespace logger

#endif /* UR_LEVEL_HPP */
