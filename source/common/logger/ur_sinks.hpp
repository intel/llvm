// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef UR_SINKS_HPP
#define UR_SINKS_HPP 1

#include <fstream>
#include <iostream>
#include <sstream>

#include "ur_filesystem_resolved.hpp"
#include "ur_level.hpp"

namespace logger {

class Sink {
  public:
    template <typename... Args>
    void log(logger::Level level, const char *fmt, Args &&...args) {
        if (!skip_prefix) {
            *ostream << "<" << logger_name << ">";
            *ostream << "[" << level_to_str(level) << "]: ";
        }
        format(fmt, std::forward<Args &&>(args)...);
        *ostream << "\n";
        if (level >= flush_level) {
            ostream->flush();
        }
    }

    void setFlushLevel(logger::Level level) { this->flush_level = level; }

    virtual ~Sink() = default;

  protected:
    std::ostream *ostream;
    logger::Level flush_level;

    Sink(std::string logger_name, bool skip_prefix = false)
        : logger_name(std::move(logger_name)), skip_prefix(skip_prefix) {
        ostream = nullptr;
        flush_level = logger::Level::ERR;
    }

  private:
    std::string logger_name;
    bool skip_prefix;

    void format(const char *fmt) {
        while (*fmt != '\0') {
            while (*fmt != '{' && *fmt != '}' && *fmt != '\0') {
                *ostream << *fmt++;
            }

            if (*fmt == '{') {
                if (*(++fmt) == '{') {
                    *ostream << *fmt++;
                } else {
                    throw std::runtime_error(
                        "No arguments provided and braces not escaped!");
                }
            } else if (*fmt == '}') {
                if (*(++fmt) == '}') {
                    *ostream << *fmt++;
                } else {
                    throw std::runtime_error(
                        "Closing curly brace not escaped!");
                }
            }
        }
    }

    template <typename Arg, typename... Args>
    void format(const char *fmt, Arg &&arg, Args &&...args) {
        bool arg_printed = false;
        while (!arg_printed) {
            while (*fmt != '{' && *fmt != '}' && *fmt != '\0') {
                *ostream << *fmt++;
            }

            if (*fmt == '{') {
                if (*(++fmt) == '{') {
                    *ostream << *fmt++;
                } else if (*fmt != '}') {
                    throw std::runtime_error("Only empty braces are allowed!");
                } else {
                    *ostream << arg;
                    arg_printed = true;
                }
            } else if (*fmt == '}') {
                if (*(++fmt) == '}') {
                    *ostream << *fmt++;
                } else {
                    throw std::runtime_error(
                        "Closing curly brace not escaped!");
                }
            }
        }

        format(++fmt, std::forward<Args &&>(args)...);
    }
};

class StdoutSink : public Sink {
  public:
    StdoutSink(std::string logger_name, bool skip_prefix = false)
        : Sink(std::move(logger_name), skip_prefix) {
        this->ostream = &std::cout;
    }

    StdoutSink(std::string logger_name, Level flush_lvl,
               bool skip_prefix = false)
        : StdoutSink(std::move(logger_name), skip_prefix) {
        this->flush_level = flush_lvl;
    }

    ~StdoutSink() = default;
};

class StderrSink : public Sink {
  public:
    StderrSink(std::string logger_name, bool skip_prefix = false)
        : Sink(std::move(logger_name), skip_prefix) {
        this->ostream = &std::cerr;
    }

    StderrSink(std::string logger_name, Level flush_lvl, bool skip_prefix)
        : StderrSink(std::move(logger_name), skip_prefix) {
        this->flush_level = flush_lvl;
    }

    ~StderrSink() = default;
};

class FileSink : public Sink {
  public:
    FileSink(std::string logger_name, filesystem::path file_path,
             bool skip_prefix = false)
        : Sink(std::move(logger_name), skip_prefix) {
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
             Level flush_lvl, bool skip_prefix = false)
        : FileSink(std::move(logger_name), std::move(file_path), skip_prefix) {
        this->flush_level = flush_lvl;
    }

  private:
    std::ofstream ofstream;
};

inline std::unique_ptr<Sink> sink_from_str(std::string logger_name,
                                           std::string name,
                                           std::filesystem::path file_path = "",
                                           bool skip_prefix = false) {
    if (name == "stdout" && file_path.empty()) {
        return std::make_unique<logger::StdoutSink>(logger_name, skip_prefix);
    } else if (name == "stderr" && file_path.empty()) {
        return std::make_unique<logger::StderrSink>(logger_name, skip_prefix);
    } else if (name == "file" && !file_path.empty()) {
        return std::make_unique<logger::FileSink>(logger_name, file_path,
                                                  skip_prefix);
    }

    throw std::invalid_argument(
        std::string("Parsing error: no valid sink for string '") + name +
        std::string("' with path '") + file_path.string() + std::string("'.") +
        std::string("\nValid sink names are: stdout, stderr, file"));
}

} // namespace logger

#endif /* UR_SINKS_HPP */
