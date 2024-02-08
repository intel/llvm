// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_LOGGER_DETAILS_HPP
#define UR_LOGGER_DETAILS_HPP 1

#include "ur_level.hpp"
#include "ur_sinks.hpp"

namespace logger {

struct LegacyMessage {
    LegacyMessage(const char *p) : message(p){};
    const char *message;
};

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
    void debug(const logger::LegacyMessage &p, const char *format,
               Args &&...args) {
        log(p, logger::Level::DEBUG, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void info(const logger::LegacyMessage &p, const char *format,
              Args &&...args) {
        log(p, logger::Level::INFO, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void warning(const logger::LegacyMessage &p, const char *format,
                 Args &&...args) {
        log(p, logger::Level::WARN, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void error(const logger::LegacyMessage &p, const char *format,
               Args &&...args) {
        log(p, logger::Level::ERR, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void log(logger::Level level, const char *format, Args &&...args) {
        log(logger::LegacyMessage(format), level, format,
            std::forward<Args>(args)...);
    }

    template <typename... Args>
    void log(const logger::LegacyMessage &p, logger::Level level,
             const char *format, Args &&...args) {
        if (!sink) {
            return;
        }

        if (isLegacySink) {
            sink->log(level, p.message, std::forward<Args>(args)...);
            return;
        }
        if (level < this->level) {
            return;
        }

        sink->log(level, format, std::forward<Args>(args)...);
    }

    void setLegacySink(std::unique_ptr<logger::Sink> legacySink) {
        this->isLegacySink = true;
        this->sink = std::move(legacySink);
    }

  private:
    logger::Level level;
    std::unique_ptr<logger::Sink> sink;
    bool isLegacySink = false;
};

} // namespace logger

#endif /* UR_LOGGER_DETAILS_HPP */
