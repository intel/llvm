/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
 * LLVM-exception
 *
 * @file test_collector.cpp
 *
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

XPTI_CALLBACK_API void trace_cb(uint16_t trace_type, xpti::trace_event_data_t *,
                                xpti::trace_event_data_t *child, uint64_t,
                                const void *user_data) {
  auto *args = static_cast<const xpti::function_with_args_t *>(user_data);
  auto *payload = xptiQueryPayload(child);
  std::cerr << (trace_type == TRACE_FN_BEGIN ? "begin" : "end");
  std::cerr << " " << args->function_name << " " << args->function_id;
  if (payload) {
    std::cerr << " " << payload->name << " " << payload->source_file << " "
              << payload->line_no << " " << payload->column_no;
  }
  std::cerr << std::endl;
}

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

  xptiRegisterCallback(stream_id, TRACE_FN_BEGIN, trace_cb);
  xptiRegisterCallback(stream_id, TRACE_FN_END, trace_cb);
}

XPTI_CALLBACK_API void xptiTraceFinish(const char *) { /* noop */ }
