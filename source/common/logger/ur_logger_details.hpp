// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef UR_LOGGER_DETAILS_HPP
#define UR_LOGGER_DETAILS_HPP 1

#include "ur_level.hpp"
#include "ur_sinks.hpp"

namespace logger {

class Logger {
  public:
    Logger(std::unique_ptr<logger::Sink> sink) : sink(std::move(sink)) {
        if (!this->sink) {
            throw std::invalid_argument(
                "Can't create logger with nullptr Sink!");
        }
        this->level = logger::Level::QUIET;
    }

    Logger(logger::Level level, std::unique_ptr<logger::Sink> sink)
        : level(level), sink(std::move(sink)) {
        if (!this->sink) {
            throw std::invalid_argument(
                "Can't create logger with nullptr Sink!");
        }
    }

    Logger(const Logger &other) : level(other.level), sink(std::move(sink)) {}

    Logger &operator=(Logger other) {
        std::swap(level, other.level);
        sink.swap(other.sink);
        return *this;
    }

    ~Logger() = default;

    void setLevel(logger::Level level) { this->level = level; }

    void setFlushLevel(logger::Level level) {
        this->sink->setFlushLevel(level);
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

    template <typename... Args> void error(const char *format, Args &&...args) {
        log(logger::Level::ERR, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void log(logger::Level level, const char *format, Args &&...args) {
        if (level < this->level) {
            return;
        }

        sink->log(level, format, std::forward<Args>(args)...);
    }

  private:
    logger::Level level;
    std::unique_ptr<logger::Sink> sink;
};

} // namespace logger

#endif /* UR_LOGGER_DETAILS_HPP */
