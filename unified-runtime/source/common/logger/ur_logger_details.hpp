// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_LOGGER_DETAILS_HPP
#define UR_LOGGER_DETAILS_HPP 1

#include "ur_level.hpp"
#include "ur_sinks.hpp"

namespace logger {

struct LegacyMessage {
  LegacyMessage(const char *p) : message(p) {};
  const char *message;
};

class Logger {
public:
  Logger(std::unique_ptr<logger::Sink> sink,
         std::unique_ptr<logger::CallbackSink> callbackSink = nullptr)
      : standardSinkLevel(UR_LOGGER_LEVEL_QUIET), standardSink(std::move(sink)),
        callbackSinkLevel(UR_LOGGER_LEVEL_QUIET),
        callbackSink(std::move(callbackSink)) {}

  Logger(ur_logger_level_t level = UR_LOGGER_LEVEL_QUIET,
         std::unique_ptr<logger::Sink> sink = nullptr,
         ur_logger_level_t callbackSinkLevel = UR_LOGGER_LEVEL_QUIET,
         std::unique_ptr<logger::CallbackSink> callbackSink = nullptr)
      : standardSinkLevel(level), standardSink(std::move(sink)),
        callbackSinkLevel(callbackSinkLevel),
        callbackSink(std::move(callbackSink)) {}

  void setLevel(ur_logger_level_t level) { this->standardSinkLevel = level; }

  ur_logger_level_t getLevel() { return this->standardSinkLevel; }

  void setFlushLevel(ur_logger_level_t level) {
    if (standardSink) {
      this->standardSink->setFlushLevel(level);
    }
  }

  template <typename... Args>
  void log(ur_logger_level_t level, const char *filename, const char *lineno,
           const char *format, Args &&...args) {
    log(logger::LegacyMessage(format), level, filename, lineno, format,
        std::forward<Args>(args)...);
  }

  template <typename... Args>
  void log(const logger::LegacyMessage &p, ur_logger_level_t level,
           const char *filename, const char *lineno, const char *format,
           Args &&...args) {
    if (callbackSink && level >= this->callbackSinkLevel) {
      callbackSink->log(level, filename, lineno, format,
                        std::forward<Args>(args)...);
    }

    if (standardSink) {
      if (isLegacySink) {
        standardSink->log(level, filename, lineno, p.message,
                          std::forward<Args>(args)...);
        return;
      }

      if (level < this->standardSinkLevel) {
        return;
      }
      standardSink->log(level, filename, lineno, format,
                        std::forward<Args>(args)...);
    }
  }

  void setLegacySink(std::unique_ptr<Sink> legacySink) {
    this->isLegacySink = true;
    this->standardSink = std::move(legacySink);
  }

  void setCallbackSink(ur_logger_callback_t callBack, void *pUserData,
                       ur_logger_level_t level = UR_LOGGER_LEVEL_QUIET) {
    if (!callbackSink) {
      callbackSink = std::make_unique<CallbackSink>("UR_LOG_CALLBACK");
    }

    if (callBack) {
      callbackSink->setCallback(callBack, pUserData);
      callbackSinkLevel = level;
    }
  }

  void setCallbackLevel(ur_logger_level_t level) {
    this->callbackSinkLevel = level;
  }

private:
  ur_logger_level_t standardSinkLevel;
  std::unique_ptr<logger::Sink> standardSink;
  bool isLegacySink = false;

  ur_logger_level_t callbackSinkLevel;
  std::unique_ptr<logger::CallbackSink> callbackSink;
};

} // namespace logger

#ifdef SRC_PATH_SIZE
#define SHORT_FILE ((__FILE__) + (SRC_PATH_SIZE))
#else
#define SHORT_FILE __FILE__
#endif

#define UR_STRIMPL_(x) #x
#define UR_STR_(x) UR_STRIMPL_(x)

#define URLOG_(logger_instance, level, ...)                                    \
  {                                                                            \
    (logger_instance)                                                          \
        .log(UR_LOGGER_LEVEL_##level, SHORT_FILE, UR_STR_(__LINE__),           \
             __VA_ARGS__);                                                     \
  }

#define URLOG_ALWAYS_(logger_instance, ...)                                    \
  URLOG_(logger_instance, QUIET, __VA_ARGS__)

#define URLOG_LEGACY_(logger_instance, level, legacy_message, ...)             \
  {                                                                            \
    (logger_instance)                                                          \
        .log(legacy_message, UR_LOGGER_LEVEL_##level, SHORT_FILE,              \
             UR_STR_(__LINE__), __VA_ARGS__);                                  \
  }

#endif /* UR_LOGGER_DETAILS_HPP */
