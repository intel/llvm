// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef UR_LOGGER_HPP
#define UR_LOGGER_HPP 1

#include "ur_logger_details.hpp"
#include "ur_util.hpp"

namespace logger {

void print_backtrace();

#define UR_FASSERT(expr, msg)                                                  \
  if (!(expr)) {                                                               \
    std::cerr << "ASSERTION FAILED at " __FILE__ ":" << __LINE__               \
              << " ((" #expr ")) " << msg << '\n';                             \
    logger::print_backtrace();                                                 \
    UR_LOG(ERR, "ASSERTION FAILED at " __FILE__ ":{} ((" #expr "))",           \
           __LINE__);                                                          \
    abort();                                                                   \
  }

#ifdef UR_DASSERT_ENABLED
#define UR_DASSERT(expr, msg) UR_FASSERT(expr, msg)
#else
#define UR_DASSERT(expr, msg)                                                  \
  {                                                                            \
    while (0) {                                                                \
    }                                                                          \
  };
#endif

// a fatal failure - always aborts program
#define UR_FFAILURE(msg) UR_FASSERT(false, msg)
// a debug failure - aborts program in debug mode or with assertions enabled
#define UR_DFAILURE(msg) UR_DASSERT(false, msg)

/// @brief Create an instance of the logger with parameters obtained from the
/// respective
///        environment variable or with default configuration if the env var is
///        empty, not set, or has the wrong format. Logger env vars are in the
///        format: UR_LOG_*, ie.:
///        - UR_LOG_LOADER (logger for loader library),
///        - UR_LOG_NULL (logger for null adapter).
///        Example of env var for setting up a loader library logger with
///        logging level set to `info`, flush level set to `warning`, and output
///        set to the `out.log` file:
///             UR_LOG_LOADER="level:info;flush:warning;output:file,out.log"
/// @param logger_name name that should be appended to the `UR_LOG_` prefix to
///        get the proper environment variable, ie. "loader"
/// @param default_log_level provides the default logging configuration when the
/// environment
///        variable is not provided or cannot be parsed
/// @return an instance of a logger::Logger. In case of failure in the parsing
/// of
///         the environment variable, returns a default logger with the
///         following options:
///             - log level: quiet, meaning no messages are printed
///             - flush level: error, meaning that only error messages are
///             guaranteed
///                            to be printed immediately as they occur
///             - output: stderr
Logger
create_logger(std::string logger_name, bool skip_prefix = false,
              bool skip_linebreak = false,
              ur_logger_level_t default_log_level = UR_LOGGER_LEVEL_QUIET);

inline Logger &
get_logger(std::string name = "common",
           ur_logger_level_t default_log_level = UR_LOGGER_LEVEL_QUIET) {
  static Logger logger =
      create_logger(std::move(name), /*skip_prefix*/ false,
                    /*slip_linebreak*/ false, default_log_level);
  return logger;
}

inline void init(const std::string &name) { get_logger(name.c_str()); }

// use log level as a first parameter
// available levels: QUIET, ERR, WARN, INFO, DEBUG
#define UR_LOG(level, ...) URLOG_(::logger::get_logger(), level, __VA_ARGS__)

// TODO: consider removing UR_LOG_L and maybe UR_LOG_LEGACY macros, using UR_LOG
// instead
#define UR_LOG_LEGACY(level, legacy_message, ...)                              \
  URLOG_L_(::logger::get_logger(), level, legacy_message, __VA_ARGS__)
#define UR_LOG_L(logger, level, ...) URLOG_(logger, level, __VA_ARGS__)

// safe version of UR_LOG that catches exceptions from the logger
#define UR_LOG_SAFE(level, ...)                                                \
  do {                                                                         \
    try {                                                                      \
      UR_LOG(level, __VA_ARGS__);                                              \
    } catch (const std::exception &e) {                                        \
      std::fprintf(stderr, "Error during logging: %s\n", e.what());            \
    } catch (...) {                                                            \
      std::fprintf(stderr, "Unknown error during logging\n");                  \
    }                                                                          \
  } while (0)

inline void setLevel(ur_logger_level_t level) { get_logger().setLevel(level); }

inline void setFlushLevel(ur_logger_level_t level) {
  get_logger().setFlushLevel(level);
}

template <typename T> std::string toHex(T &&t) {
  std::ostringstream s;
  s << std::hex << t;
  return s.str();
}

template <typename T> std::string makeStringFromStreamable(T &&obj) {
  std::ostringstream s;
  s << obj;
  return s.str();
}

} // namespace logger

#endif /* UR_LOGGER_HPP */
