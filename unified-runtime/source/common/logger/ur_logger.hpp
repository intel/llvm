// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef UR_LOGGER_HPP
#define UR_LOGGER_HPP 1

#include <algorithm>
#include <memory>

#include "ur_logger_details.hpp"
#include "ur_util.hpp"

namespace logger {

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

inline void setLevel(ur_logger_level_t level) { get_logger().setLevel(level); }

inline void setFlushLevel(ur_logger_level_t level) {
  get_logger().setFlushLevel(level);
}

template <typename T> inline std::string toHex(T t) {
  std::stringstream s;
  s << std::hex << t;
  return s.str();
}

inline bool str_to_bool(const std::string &str) {
  if (!str.empty()) {
    std::string lower_value = str;
    std::transform(lower_value.begin(), lower_value.end(), lower_value.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    const std::initializer_list<std::string> true_str = {"y", "yes", "t",
                                                         "true", "1"};
    return std::find(true_str.begin(), true_str.end(), lower_value) !=
           true_str.end();
  }

  return false;
}

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
inline Logger create_logger(std::string logger_name, bool skip_prefix,
                            bool skip_linebreak,
                            ur_logger_level_t default_log_level) {
  std::transform(logger_name.begin(), logger_name.end(), logger_name.begin(),
                 ::toupper);
  const std::string env_var_name = "UR_LOG_" + logger_name;
  const auto default_flush_level = UR_LOGGER_LEVEL_ERROR;
  const std::string default_output = "stderr";
  const bool default_fileline = false;
  auto flush_level = default_flush_level;
  ur_logger_level_t level = default_log_level;
  bool fileline = default_fileline;
  std::unique_ptr<Sink> sink;

  try {
    auto map = getenv_to_map(env_var_name.c_str());
    if (!map.has_value()) {
      return Logger(default_log_level,
                    std::make_unique<logger::StderrSink>(
                        std::move(logger_name), skip_prefix, skip_linebreak));
    }

    auto kv = map->find("level");
    if (kv != map->end()) {
      auto value = kv->second.front();
      level = str_to_level(std::move(value));
      map->erase(kv);
    }

    kv = map->find("flush");
    if (kv != map->end()) {
      auto value = kv->second.front();
      flush_level = str_to_level(std::move(value));
      map->erase(kv);
    }

    kv = map->find("fileline");
    if (kv != map->end()) {
      auto value = kv->second.front();
      fileline = str_to_bool(std::move(value));
      map->erase(kv);
    }

    std::vector<std::string> values = {default_output};
    kv = map->find("output");
    if (kv != map->end()) {
      values = kv->second;
      map->erase(kv);
    }

    if (!map->empty()) {
      std::cerr << "Wrong logger environment variable parameter: '"
                << map->begin()->first << "'. Default logger options are set.";
      return Logger(default_log_level,
                    std::make_unique<logger::StderrSink>(
                        std::move(logger_name), skip_prefix, skip_linebreak));
    }

    sink = values.size() == 2 ? sink_from_str(logger_name, values[0], values[1],
                                              skip_prefix, skip_linebreak)
                              : sink_from_str(logger_name, values[0], "",
                                              skip_prefix, skip_linebreak);
  } catch (const std::invalid_argument &e) {
    std::cerr << "Error when creating a logger instance from the '"
              << env_var_name << "' environment variable:\n"
              << e.what() << std::endl;
    return Logger(default_log_level,
                  std::make_unique<logger::StderrSink>(
                      std::move(logger_name), skip_prefix, skip_linebreak));
  }

  sink->setFlushLevel(flush_level);
  sink->setFileLine(fileline);

  return Logger(level, std::move(sink));
}

} // namespace logger

#endif /* UR_LOGGER_HPP */
