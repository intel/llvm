/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file collector.cpp
 *
 * This file contains the implementation of the UR collector library. The library
 * provides instrumentation for tracing function calls and profiling function
 * execution time.
 */

#include <chrono>
#include <optional>
#include <regex>
#include <sstream>
#include <stack>
#include <string>
#include <string_view>
#include <vector>

#include "logger/ur_logger.hpp"
#include "ur_api.h"
#include "ur_params.hpp"
#include "xpti/xpti_trace_framework.h"

constexpr uint16_t TRACE_FN_BEGIN =
    static_cast<uint16_t>(xpti::trace_point_type_t::function_with_args_begin);
constexpr uint16_t TRACE_FN_END =
    static_cast<uint16_t>(xpti::trace_point_type_t::function_with_args_end);
constexpr std::string_view UR_STREAM_NAME = "ur";

logger::Logger out = logger::create_logger("collector", true);

constexpr const char *ARGS_ENV = "UR_COLLECTOR_ARGS";

enum time_unit {
    TIME_UNIT_AUTO,
    TIME_UNIT_NS,
    TIME_UNIT_US,
    TIME_UNIT_MS,
    TIME_UNIT_S,
    MAX_TIME_UNIT,
};

const char *time_unit_str[MAX_TIME_UNIT] = {"auto", "ns", "us", "ms", "s"};

std::string time_to_str(std::chrono::nanoseconds dur, enum time_unit unit) {
    std::ostringstream ostr;

    switch (unit) {
    case TIME_UNIT_AUTO: {
        if (dur.count() < 1000) {
            return time_to_str(dur, TIME_UNIT_NS);
        }
        if (dur.count() < 1000 * 1000) {
            return time_to_str(dur, TIME_UNIT_US);
        }
        if (dur.count() < 1000 * 1000 * 1000) {
            return time_to_str(dur, TIME_UNIT_MS);
        }
        return time_to_str(dur, TIME_UNIT_S);
    } break;
    case TIME_UNIT_NS: {
        ostr << dur.count() << "ns";
    } break;
    case TIME_UNIT_US: {
        std::chrono::duration<double, std::micro> d = dur;
        ostr << d.count() << "us";
    } break;
    case TIME_UNIT_MS: {
        std::chrono::duration<double, std::milli> d = dur;
        ostr << d.count() << "ms";
    } break;
    case TIME_UNIT_S: {
        std::chrono::duration<double, std::ratio<1>> d = dur;
        ostr << d.count() << "s";
    } break;
    default: {
        out.error("invalid time unit {}", unit);
        break;
    }
    }

    return ostr.str();
}

/*
 * Since this is a library that gets loaded alongside the traced program, it
 * can't just accept arguments from the trace CLI tool directly. Instead, the
 * arguments are passed through an environment variable. Users shouldn't set
 * these directly.
 *
 * Available arguments (documented in trace cli tool):
 * - "print_begin"
 * - "profiling"
 * - "time_unit:<auto,ns, ...>"
 * - "filter:<regex>"
 */
static class cli_args {
    std::optional<std::string>
    arg_with_value(std::string_view name, const std::string arg_name,
                   const std::vector<std::string> arg_values) {
        if (arg_name != name) {
            return std::nullopt;
        }
        if (arg_values.size() != 1) {
            out.warn("{} requires a single argument, skipping...", name);
            return std::nullopt;
        }
        return arg_values.at(0);
    }

  public:
    cli_args() {
        print_begin = false;
        profiling = false;
        time_unit = TIME_UNIT_AUTO;
        no_args = false;
        filter = std::nullopt;
        filter_str = std::nullopt;
        if (auto args = getenv_to_map(ARGS_ENV, false)) {
            for (auto [arg_name, arg_values] : *args) {
                if (arg_name == "print_begin") {
                    print_begin = true;
                } else if (arg_name == "profiling") {
                    profiling = true;
                } else if (arg_name == "no_args") {
                    no_args = true;
                } else if (auto unit = arg_with_value("time_unit", arg_name,
                                                      arg_values)) {
                    for (int i = 0; i < MAX_TIME_UNIT; ++i) {
                        if (time_unit_str[i] == unit) {
                            time_unit = (enum time_unit)i;
                            break;
                        }
                    }
                } else if (auto filter_str =
                               arg_with_value("filter", arg_name, arg_values)) {
                    try {
                        filter = filter_str;
                    } catch (const std::regex_error &err) {
                        out.warn("invalid filter regex {} {}", *filter_str,
                                 err.what());
                    }
                } else {
                    out.warn("unknown {} argument {}.", ARGS_ENV, arg_name);
                }
            }
        }
        out.debug("collector args (.print_begin = {}, .profiling = {}, "
                  ".time_unit = {}, .filter = {})",
                  print_begin, profiling, time_unit_str[time_unit],
                  filter_str.has_value() ? *filter_str : "none");
    }

    enum time_unit time_unit;
    bool print_begin;
    bool profiling;
    bool no_args;
    std::optional<std::string>
        filter_str; //the filter_str is kept primarly for printing.
    std::optional<std::regex> filter;
} cli_args;

using namespace ur_params;

typedef std::chrono::high_resolution_clock Clock;

struct fn_context {
    uint64_t instance;
    std::optional<std::chrono::time_point<Clock>> start;
};

static thread_local std::stack<fn_context> instance_data;

fn_context *push_instance_data(uint64_t instance) {
    instance_data.push(fn_context{instance, std::nullopt});
    return &instance_data.top();
}

std::optional<fn_context> pop_instance_data(uint64_t instance) {
    if (instance_data.empty()) {
        return std::nullopt;
    }
    auto data = instance_data.top();
    if (data.instance != instance) {
        return std::nullopt;
    }
    instance_data.pop();
    return data;
}

void trace_begin(const xpti::function_with_args_t *args, uint64_t instance,
                 fn_context *ctx) {
    if (cli_args.print_begin) {
        std::ostringstream args_str;
        if (cli_args.no_args) {
            args_str << "...";
        } else {
            ur_params::serializeFunctionParams(args_str, args->function_id,
                                               args->args_data);
        }
        out.info("begin({}) - {}({});", instance, args->function_name,
                 args_str.str());
    }
    // start the clock as the very last thing this function does to minimize
    // tracing overheads
    if (cli_args.profiling) {
        ctx->start = Clock::now();
    }
}

void trace_end(const xpti::function_with_args_t *args, uint64_t instance,
               fn_context ctx, std::chrono::time_point<Clock> time) {
    std::ostringstream args_str;
    if (cli_args.no_args) {
        args_str << "...";
    } else {
        ur_params::serializeFunctionParams(args_str, args->function_id,
                                           args->args_data);
    }

    std::ostringstream prefix_str;
    if (cli_args.print_begin) {
        prefix_str << "end(" << instance << ") - ";
    }
    auto result = static_cast<const ur_result_t *>(args->ret_data);

    std::ostringstream profile_str;
    if (ctx.start && cli_args.profiling) {
        auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(
            time - *ctx.start);
        profile_str << " (" << time_to_str(dur, cli_args.time_unit) << ")";
    }
    out.info("{}{}({}) -> {};{}", prefix_str.str(), args->function_name,
             args_str.str(), *result, profile_str.str());
}

XPTI_CALLBACK_API void trace_cb(uint16_t trace_type,
                                xpti::trace_event_data_t *parent,
                                xpti::trace_event_data_t *event,
                                uint64_t instance, const void *user_data) {
    // stop the the clock as the very first thing, only used for TRACE_FN_END
    auto time = Clock::now();
    auto *args = static_cast<const xpti::function_with_args_t *>(user_data);

    if (auto regex = cli_args.filter) {
        if (!std::regex_match(args->function_name, *regex)) {
            out.debug("function {} does not match regex filter, skipping...",
                      args->function_name);
            return;
        }
    }

    if (trace_type == TRACE_FN_BEGIN) {
        auto ctx = push_instance_data(instance);
        trace_begin(args, instance, ctx);
    } else if (trace_type == TRACE_FN_END) {
        auto ctx = pop_instance_data(instance);
        if (!ctx) {
            out.error("Received TRACE_FN_END without corresponding "
                      "TRACE_FN_BEGIN, instance {}. Skipping...",
                      instance);
            return;
        }
        trace_end(args, instance, *ctx, time);
    } else {
        out.warn("unsupported trace type");
    }
}

/**
 * @brief Subscriber initialization function called by the XPTI dispatcher.
 *
 * Called for every stream.
 */
XPTI_CALLBACK_API void xptiTraceInit(unsigned int major_version,
                                     unsigned int minor_version,
                                     const char *version_str,
                                     const char *stream_name) {
    if (!stream_name || std::string_view(stream_name) != UR_STREAM_NAME) {
        out.debug("Found stream: {}. Expected: {}. Skipping...", stream_name,
                  UR_STREAM_NAME);
        return;
    }

    if (UR_MAKE_VERSION(major_version, minor_version) !=
        UR_API_VERSION_CURRENT) {
        out.error("Invalid stream version: {}.{}. Expected: {}.{}. Skipping...",
                  major_version, minor_version,
                  UR_MAJOR_VERSION(UR_API_VERSION_CURRENT),
                  UR_MINOR_VERSION(UR_API_VERSION_CURRENT));
        return;
    }

    uint8_t stream_id = xptiRegisterStream(stream_name);

    out.debug("Registered stream {} ({}.{}).", stream_name, major_version,
              minor_version);

    xptiRegisterCallback(stream_id, TRACE_FN_BEGIN, trace_cb);
    xptiRegisterCallback(stream_id, TRACE_FN_END, trace_cb);
}

/**
 * @brief Subscriber finish function called by the XPTI dispatcher.
 */
XPTI_CALLBACK_API void xptiTraceFinish(const char *stream_name) { /* noop */
}
