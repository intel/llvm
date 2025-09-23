/*
 *
 * Copyright (C) 2022-2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <algorithm>

#include "../backtrace.hpp"
#include "ur_logger.hpp"

namespace logger {

void print_backtrace() {
  for (auto btLine : ur::getCurrentBacktrace()) {
    std::cerr << btLine << std::endl;
  }
}

static bool str_to_bool(const std::string &str) {
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

Logger create_logger(std::string logger_name, bool skip_prefix,
                     bool skip_linebreak, ur_logger_level_t default_log_level) {
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
