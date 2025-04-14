// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_SINKS_HPP
#define UR_SINKS_HPP 1

#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>

#include "ur_api.h"
#include "ur_filesystem_resolved.hpp"
#include "ur_level.hpp"
#include "ur_print.hpp"

namespace logger {

#if defined(_WIN32)
inline bool isTearDowned = false;
#endif

class Sink {
public:
  template <typename... Args>
  void log(ur_logger_level_t level, const char *filename, const char *lineno,
           const char *fmt, Args &&...args) {
    std::ostringstream buffer;
    if (!skip_prefix && level != UR_LOGGER_LEVEL_QUIET) {
      buffer << "<" << logger_name << ">"
             << "[" << level_to_str(level) << "]: ";
    }

    format(buffer, filename, lineno, fmt, std::forward<Args &&>(args)...);
    if (add_fileline) {
      buffer << " <" << filename << ":" << lineno << ">";
    }
    if (!skip_linebreak) {
      buffer << "\n";
    }
// This is a temporary workaround on windows, where UR adapter is teardowned
// before the UR loader, which will result in access violation when we use print
// function as the overrided print function was already released with the UR
// adapter.
// TODO: Change adapters to use a common sink class in the loader instead of
// using thier own sink class that inherit from logger::Sink.
#if defined(_WIN32)
    if (isTearDowned) {
      std::cerr << buffer.str() << "\n";
    } else {
      print(level, buffer.str());
    }
#else
    print(level, buffer.str());
#endif
  }

  void setFileLine(bool fileline) { add_fileline = fileline; }
  void setFlushLevel(ur_logger_level_t level) { this->flush_level = level; }

  virtual ~Sink() = default;

protected:
  std::ostream *ostream;
  ur_logger_level_t flush_level;

  Sink(std::string logger_name, bool skip_prefix = false,
       bool skip_linebreak = false)
      : logger_name(std::move(logger_name)), skip_prefix(skip_prefix),
        skip_linebreak(skip_linebreak), add_fileline(false) {
    ostream = nullptr;
    flush_level = UR_LOGGER_LEVEL_ERROR;
  }

  virtual void print(ur_logger_level_t level, const std::string &msg) {
    std::scoped_lock<std::mutex> lock(output_mutex);
    *ostream << msg;
    if (level >= flush_level) {
      ostream->flush();
    }
  }

private:
  std::string logger_name;
  const bool skip_prefix;
  const bool skip_linebreak;
  bool add_fileline;
  std::mutex output_mutex;
  const char *error_prefix = "Log message syntax error: ";

  void format(std::ostringstream &buffer, const char *filename,
              const char *lineno, const char *fmt) {
    while (*fmt != '\0') {
      while (*fmt != '{' && *fmt != '}' && *fmt != '\0') {
        buffer << *fmt++;
      }

      if (*fmt == '{') {
        if (*(++fmt) == '{') {
          buffer << *fmt++;
        } else {
          std::cerr << error_prefix
                    << "No arguments provided and braces not escaped!"
                    << filename << ":" << lineno << std::endl;
        }
      } else if (*fmt == '}') {
        if (*(++fmt) == '}') {
          buffer << *fmt++;
        } else {
          std::cerr << error_prefix << "Closing curly brace not escaped!"
                    << filename << ":" << lineno << std::endl;
        }
      }
    }
  }

  template <typename Arg, typename... Args>
  void format(std::ostringstream &buffer, const char *filename,
              const char *lineno, const char *fmt, Arg &&arg, Args &&...args) {
    bool arg_printed = false;
    while (!arg_printed) {
      while (*fmt != '{' && *fmt != '}' && *fmt != '\0') {
        buffer << *fmt++;
      }

      if (*fmt == '{') {
        if (*(++fmt) == '{') {
          buffer << *fmt++;
        } else if (*fmt != '}') {
          std::cerr << error_prefix << "Only empty braces are allowed!"
                    << filename << ":" << lineno << std::endl;
        } else {
          buffer << arg;
          arg_printed = true;
        }
      } else if (*fmt == '}') {
        if (*(++fmt) == '}') {
          buffer << *fmt++;
        } else {
          std::cerr << error_prefix << "Closing curly brace not escaped!"
                    << filename << ":" << lineno << std::endl;
        }
      }

      if (*fmt == '\0') {
        std::cerr << error_prefix << filename << ":" << lineno
                  << "Too many arguments! first excessive:" << arg << std::endl;
        // ignore all left arguments and finalize message
        format(buffer, filename, lineno, fmt);
        return;
      }
    }

    format(buffer, filename, lineno, ++fmt, std::forward<Args &&>(args)...);
  }
};

class StdoutSink : public Sink {
public:
  StdoutSink(std::string logger_name, bool skip_prefix = false,
             bool skip_linebreak = false)
      : Sink(std::move(logger_name), skip_prefix, skip_linebreak) {
    this->ostream = &std::cout;
  }

  StdoutSink(std::string logger_name, ur_logger_level_t flush_lvl,
             bool skip_prefix = false, bool skip_linebreak = false)
      : StdoutSink(std::move(logger_name), skip_prefix, skip_linebreak) {
    this->flush_level = flush_lvl;
  }

  ~StdoutSink() = default;
};

class StderrSink : public Sink {
public:
  StderrSink(std::string logger_name, bool skip_prefix = false,
             bool skip_linebreak = false)
      : Sink(std::move(logger_name), skip_prefix, skip_linebreak) {
    this->ostream = &std::cerr;
  }

  StderrSink(std::string logger_name, ur_logger_level_t flush_lvl,
             bool skip_prefix, bool skip_linebreak)
      : StderrSink(std::move(logger_name), skip_prefix, skip_linebreak) {
    this->flush_level = flush_lvl;
  }

  ~StderrSink() = default;
};

class FileSink : public Sink {
public:
  FileSink(std::string logger_name, filesystem::path file_path,
           bool skip_prefix = false, bool skip_linebreak = false)
      : Sink(std::move(logger_name), skip_prefix, skip_linebreak) {
    ofstream = std::ofstream(file_path);
    if (!ofstream.good()) {
      std::stringstream ss;
      ss << "Failure while opening log file " << file_path.string()
         << ". Check if given path exists.";
      throw std::invalid_argument(ss.str());
    }
    this->ostream = &ofstream;
  }

  FileSink(std::string logger_name, filesystem::path file_path,
           ur_logger_level_t flush_lvl, bool skip_prefix = false,
           bool skip_linebreak = false)
      : FileSink(std::move(logger_name), std::move(file_path), skip_prefix,
                 skip_linebreak) {
    this->flush_level = flush_lvl;
  }

  ~FileSink() = default;

private:
  std::ofstream ofstream;
};

class CallbackSink : public Sink {
public:
  CallbackSink(std::string logger_name, bool skip_prefix = false,
               bool skip_linebreak = false)
      : Sink(std::move(logger_name), skip_prefix, skip_linebreak) {}

  CallbackSink(std::string logger_name, ur_logger_level_t flush_lvl,
               bool skip_prefix, bool skip_linebreak)
      : CallbackSink(std::move(logger_name), skip_prefix, skip_linebreak) {
    this->flush_level = flush_lvl;
  }

  ~CallbackSink() = default;

  void setCallback(ur_logger_callback_t cb, void *pUserData) {
    callback = cb;
    userData = pUserData;
  }

private:
  ur_logger_callback_t callback = nullptr;
  void *userData = nullptr;

  virtual void print(ur_logger_level_t level, const std::string &msg) override {
    if (callback) {
      callback(level, msg.c_str(), userData);
    }
  }
};

inline std::unique_ptr<Sink> sink_from_str(std::string logger_name,
                                           std::string name,
                                           filesystem::path file_path = "",
                                           bool skip_prefix = false,
                                           bool skip_linebreak = false) {
  if (name == "stdout" && file_path.empty()) {
    return std::make_unique<StdoutSink>(logger_name, skip_prefix,
                                        skip_linebreak);
  } else if (name == "stderr" && file_path.empty()) {
    return std::make_unique<StderrSink>(logger_name, skip_prefix,
                                        skip_linebreak);
  } else if (name == "file" && !file_path.empty()) {
    return std::make_unique<FileSink>(logger_name, file_path, skip_prefix,
                                      skip_linebreak);
  }

  throw std::invalid_argument(
      std::string("Parsing error: no valid sink for string '") + name +
      std::string("' with path '") + file_path.string() + std::string("'.") +
      std::string("\nValid sink names are: stdout, stderr, file"));
}

} // namespace logger

#endif /* UR_SINKS_HPP */
