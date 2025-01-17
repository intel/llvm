/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
 * LLVM-exception
 *
 * @file collector.cpp
 *
 * @brief UR collector library example for use with the XPTI framework.
 *
 * The collector example demonstrates the use of loader's tracing functionality
 * that's integrated with the XPTI framework.
 * This example can be loaded into any UR-based software and will trace all
 * UR calls and print information about each to standard output.
 */

#include <cassert>
#include <iostream>
#include <iterator>
#include <memory>
#include <ostream>
#include <sstream>
#include <string_view>

#include "ur_api.h"
#include "xpti/xpti_trace_framework.h"

constexpr uint16_t TRACE_FN_BEGIN =
    static_cast<uint16_t>(xpti::trace_point_type_t::function_with_args_begin);
constexpr uint16_t TRACE_FN_END =
    static_cast<uint16_t>(xpti::trace_point_type_t::function_with_args_end);
constexpr std::string_view UR_STREAM_NAME = "ur.call";

/**
 * @brief Formats the function parameters and arguments for urAdapterGet
 */
std::ostream &operator<<(std::ostream &os,
                         const struct ur_adapter_get_params_t *params) {
  os << ".NumEntries = ";
  os << *params->pNumEntries;
  os << ", ";
  os << ".phAdapters = ";
  os << *params->pphAdapters;
  if (*params->pphAdapters) {
    os << " (" << **params->pphAdapters << ")";
  }
  os << ", ";
  os << ".pNumAdapters = ";
  os << *params->ppNumAdapters;
  if (*params->ppNumAdapters) {
    os << " (" << **params->ppNumAdapters << ")";
  }
  os << "";
  return os;
}

/**
 * A map of functions that format the parameters and arguments for each UR
 * function. This example only implements a handler for one function,
 * `urAdapterGet`, but it's trivial to expand it to support more.
 */
static std::unordered_map<
    std::string_view,
    std::function<void(const xpti::function_with_args_t *, std::ostream &)>>
    handlers = {{"urAdapterGet", [](const xpti::function_with_args_t *fn_args,
                                    std::ostream &os) {
                   auto params =
                       static_cast<const struct ur_adapter_get_params_t *>(
                           fn_args->args_data);
                   os << params;
                 }}};

/**
 * @brief Tracing callback invoked by the dispatcher on every event.
 *
 * This function handles the incoming events and prints simple information about
 * them. Specifically, its called twice for every UR function call:
 * once before (`function_with_args_begin`) and once after
 * (`function_with_args_end`) the actual UR adapter is invoked.
 * On begin, it prints the function declaration with the call arguments
 * specified, and on end it prints the function name with the result of the
 * call.
 */
XPTI_CALLBACK_API void trace_cb(uint16_t trace_type, xpti::trace_event_data_t *,
                                xpti::trace_event_data_t *, uint64_t instance,
                                const void *user_data) {
  auto *args = static_cast<const xpti::function_with_args_t *>(user_data);
  std::ostringstream out;
  if (trace_type == TRACE_FN_BEGIN) {
    out << "function_with_args_begin(" << instance << ") - "
        << args->function_name << "(";
    auto it = handlers.find(args->function_name);
    if (it == handlers.end()) {
      out << "unimplemented";
    } else {
      it->second(args, out);
    }
    out << ");";
  } else if (trace_type == TRACE_FN_END) {
    auto result = static_cast<const ur_result_t *>(args->ret_data);
    out << "function_with_args_end(" << instance << ") - "
        << args->function_name << "(...) -> ur_result_t(" << *result << ");";
  } else {
    out << "unsupported trace type";
  }
  out << std::endl;

  std::cout << out.str();
}

/**
 * @brief Subscriber initialization function called by the XPTI dispatcher.
 *
 * This function is called for every available event stream in the traced
 * software, and enables the subscribers to register for receiving event
 * notifications for selected trace types.
 */
XPTI_CALLBACK_API void xptiTraceInit(unsigned int major_version,
                                     unsigned int minor_version, const char *,
                                     const char *stream_name) {
  if (stream_name == nullptr) {
    std::cout << "Stream name not provided. Aborting." << std::endl;
    return;
  }
  if (std::string_view(stream_name) != UR_STREAM_NAME) {
    // we expect ur.call, but this can also be xpti.framework.
    return;
  }

  if (UR_MAKE_VERSION(major_version, minor_version) != UR_API_VERSION_CURRENT) {
    std::cout << "Invalid stream version: " << major_version << "."
              << minor_version << ". Expected "
              << UR_MAJOR_VERSION(UR_API_VERSION_CURRENT) << "."
              << UR_MINOR_VERSION(UR_API_VERSION_CURRENT) << ". Aborting."
              << std::endl;
    return;
  }

  uint8_t stream_id = xptiRegisterStream(stream_name);

  /**
   * UR only issues two types of events, `function_with_args_begin` and
   * `function_with_args_end`. In this example, we handle both with the same
   * callback function.
   */
  xptiRegisterCallback(stream_id, TRACE_FN_BEGIN, trace_cb);
  xptiRegisterCallback(stream_id, TRACE_FN_END, trace_cb);
}

/**
 * @brief Subscriber finish function called by the XPTI dispatcher.
 *
 * Can be used to cleanup state or resources.
 */
XPTI_CALLBACK_API void xptiTraceFinish(const char *) { /* noop */ }
