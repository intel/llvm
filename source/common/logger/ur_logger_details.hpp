// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_LOGGER_DETAILS_HPP
#define UR_LOGGER_DETAILS_HPP 1

#include "ur_level.hpp"
#include "ur_sinks.hpp"

namespace logger {

class Logger {
  public:
    Logger(std::unique_ptr<logger::Sink> sink) : sink(std::move(sink)) {
        this->level = logger::Level::QUIET;
    }

    Logger(logger::Level level, std::unique_ptr<logger::Sink> sink)
        : level(level), sink(std::move(sink)) {}

    ~Logger() = default;

    void setLevel(logger::Level level) { this->level = level; }

    void setFlushLevel(logger::Level level) {
        if (sink) {
            this->sink->setFlushLevel(level);
        }
    }

    template <typename... Args> void debug(const char *format, Args &&...args) {
        log(logger::Level::DEBUG, format, std::forward<Args>(args)...);
    }

    template <typename... Args> void info(const char *format, Args &&...args) {
        log(logger::Level::INFO, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void warning(const char *format, Args &&...args) {
        log(logger::Level::WARN, format, std::forward<Args>(args)...);
    }

    template <typename... Args> void warn(const char *format, Args &&...args) {
        warning(format, std::forward<Args>(args)...);
    }

    template <typename... Args> void error(const char *format, Args &&...args) {
        log(logger::Level::ERR, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void always(const char *format, Args &&...args) {
        if (sink) {
            sink->log(logger::Level::QUIET, format,
                      std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    void log(logger::Level level, const char *format, Args &&...args) {
        if (level < this->level) {
            return;
        }

        if (sink) {
            sink->log(level, format, std::forward<Args>(args)...);
        }
    }

  private:
    logger::Level level;
    std::unique_ptr<logger::Sink> sink;
};

} // namespace logger

#endif /* UR_LOGGER_DETAILS_HPP */
