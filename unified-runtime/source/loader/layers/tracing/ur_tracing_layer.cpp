/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_tracing_layer.cpp
 *
 */
#include "ur_tracing_layer.hpp"
#include "ur_api.h"
#include "ur_util.hpp"
#include "xpti/xpti_data_types.h"
#include "xpti/xpti_trace_framework.h"
#include <atomic>
#include <cstdint>
#include <optional>
#include <sstream>

namespace ur_tracing_layer {
context_t *getContext() { return context_t::get_direct(); }

constexpr auto CALL_STREAM_NAME = "ur.call";
constexpr auto DEBUG_CALL_STREAM_NAME = "ur.call.debug";
constexpr auto STREAM_VER_MAJOR = UR_MAJOR_VERSION(UR_API_VERSION_CURRENT);
constexpr auto STREAM_VER_MINOR = UR_MINOR_VERSION(UR_API_VERSION_CURRENT);

// UR loader can be inited and teardown'ed multiple times in a single process.
// Unfortunately this doesn't match the semantics of XPTI, which can be
// initialized and finalized exactly once. To workaround this, XPTI is globally
// initialized on first use and finalized in the destructor.
struct XptiContextManager {
  XptiContextManager() { xptiFrameworkInitialize(); }
  ~XptiContextManager() { xptiFrameworkFinalize(); }
};

static std::shared_ptr<XptiContextManager> xptiContextManagerGet() {
  static auto contextManager = std::make_shared<XptiContextManager>();
  return contextManager;
}

// The Unified Runtime API calls are meant to be performant and creating an
// event for each API Call will add significant overheads.
static xpti_td *GURCallEvent = nullptr;
static thread_local xpti_td *activeEvent;

///////////////////////////////////////////////////////////////////////////////
context_t::context_t() : logger(logger::create_logger("tracing", true, true)) {
  this->xptiContextManager = xptiContextManagerGet();

  call_stream_id = xptiRegisterStream(CALL_STREAM_NAME);
  debug_call_stream_id = xptiRegisterStream(DEBUG_CALL_STREAM_NAME);
  std::ostringstream streamv;
  streamv << STREAM_VER_MAJOR << "." << STREAM_VER_MINOR;
  xptiInitialize(CALL_STREAM_NAME, STREAM_VER_MAJOR, STREAM_VER_MINOR,
                 streamv.str().data());
  xptiInitialize(DEBUG_CALL_STREAM_NAME, STREAM_VER_MAJOR, STREAM_VER_MINOR,
                 streamv.str().data());
  // Create global event for all UR API calls.
  xpti_tracepoint_t *Event =
      xptiCreateTracepoint("Unified Runtime call", nullptr, 0, 0, (void *)this);
  // For function_begin/function_end class of notification, the parent and the
  // event object can be NULL based on the specification.
  GURCallEvent = Event ? Event->event_ref() : nullptr;
}

void context_t::notify(uint16_t trace_type, uint32_t id, const char *name,
                       void *args, ur_result_t *resultp, uint64_t instance) {
  xpti::function_with_args_t payload{id, name, args, resultp, nullptr};
  if (xptiCheckTraceEnabled(debug_call_stream_id)) {
    xptiNotifySubscribers(debug_call_stream_id, trace_type, nullptr,
                          activeEvent, instance, &payload);
  } else {
    // Use global event for all UR API calls
    xptiNotifySubscribers(call_stream_id, trace_type, nullptr, activeEvent,
                          instance, &payload);
  }
}

uint64_t context_t::notify_begin(uint32_t id, const char *name, void *args) {
  if (xptiCheckTraceEnabled(debug_call_stream_id)) {
    // Create a new tracepoint with code location info for each UR API call.
    // This adds significant overhead to the tracing toolchain, so do this only
    // if there are debug stream subscribers.
    if (auto loc = codelocData.get_codeloc()) {
      xpti_tracepoint_t *Event = xptiCreateTracepoint(
          loc->functionName, loc->sourceFile, loc->lineNumber,
          loc->columnNumber, (void *)this);
      activeEvent = Event ? Event->event_ref() : nullptr;
    }
  } else if (xptiCheckTraceEnabled(call_stream_id)) {
    // Otherwise use global event for all UR API calls.
    activeEvent = GURCallEvent;
  } else {
    // We use UINT64_MAX as a special value that means "tracing disabled",
    // so that we don't have to repeat this check in notify_end.
    return UINT64_MAX;
  }
  uint64_t instance = xptiGetUniqueId();
  notify((uint16_t)xpti::trace_point_type_t::function_with_args_begin, id, name,
         args, nullptr, instance);
  return instance;
}

void context_t::notify_end(uint32_t id, const char *name, void *args,
                           ur_result_t *resultp, uint64_t instance) {
  if (instance == UINT64_MAX) { // tracing disabled
    return;
  }

  notify((uint16_t)xpti::trace_point_type_t::function_with_args_end, id, name,
         args, resultp, instance);
}

///////////////////////////////////////////////////////////////////////////////
context_t::~context_t() { xptiFinalize(CALL_STREAM_NAME); }
} // namespace ur_tracing_layer
