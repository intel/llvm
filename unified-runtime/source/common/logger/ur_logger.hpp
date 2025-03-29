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

Logger create_logger(std::string logger_name, bool skip_prefix = false,
                     bool skip_linebreak = false,
                     Level default_log_level = Level::QUIET);

inline Logger &get_logger(std::string name = "common",
                          Level default_log_level = Level::QUIET) {
  static Logger logger =
      create_logger(std::move(name), /*skip_prefix*/ false,
                    /*slip_linebreak*/ false, default_log_level);
  return logger;
}

inline void init(const std::string &name) { get_logger(name.c_str()); }

#define URLOG(...) URLOG_(logger::get_logger(), __VA_ARGS__)
#define URLOG_ALWAYS(...) URLOG_ALWAYS_(logger::get_logger(), __VA_ARGS__)
#define URLOG_CTX(...) URLOG_(getContext()->logger, __VA_ARGS__)
#define URLOG_CTX_ALWAYS(...) URLOG_(getContext()->logger, QUIET, __VA_ARGS__)
#define URLOG_L(...) URLOG_LEGACY_(logger::get_logger(), __VA_ARGS__)

inline void setLevel(Level level) { get_logger().setLevel(level); }

inline void setFlushLevel(Level level) { get_logger().setFlushLevel(level); }

template <typename T> inline std::string toHex(T t) {
  std::stringstream s;
  s << std::hex << t;
  return s.str();
}

inline bool str_to_bool(const std::string str) {
  return str == "true" || str == "TRUE" || str == "yes" || str == "YES" ||
         str == "y" || str == "Y" || str == "on" || str == "ON" || str == "1";
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
                            bool skip_linebreak, Level default_log_level) {
  std::transform(logger_name.begin(), logger_name.end(), logger_name.begin(),
                 ::toupper);
  const auto default_flush_level = Level::ERR;
  const std::string default_output = "stderr";
  const bool default_fileline = false;
  auto level = default_log_level;
  auto flush_level = default_flush_level;
  bool fileline = default_fileline;
  std::unique_ptr<Sink> sink;

  auto env_var_name = "UR_LOG_" + logger_name;
  try {
    auto map = getenv_to_map(env_var_name.c_str());
    if (!map.has_value()) {
      return Logger(default_log_level,
                    std::make_unique<StderrSink>(std::move(logger_name),
                                                 skip_prefix, skip_linebreak));
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

    std::vector<std::string> values = {std::move(default_output)};
    kv = map->find("output");
    if (kv != map->end()) {
      values = kv->second;
      map->erase(kv);
    }

    if (!map->empty()) {
      std::cerr << "Wrong logger environment variable parameter: '"
                << map->begin()->first << "'. Default logger options are set.";
      return Logger(default_log_level,
                    std::make_unique<StderrSink>(std::move(logger_name),
                                                 skip_prefix, skip_linebreak));
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
                  std::make_unique<StderrSink>(std::move(logger_name),
                                               skip_prefix, skip_linebreak));
  }
  sink->setFlushLevel(flush_level);
  sink->setFileLine(fileline);

  return Logger(level, std::move(sink));
}

} // namespace logger

#endif /* UR_LOGGER_HPP */
