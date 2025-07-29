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
static thread_local xpti_td *activeEvent;

///////////////////////////////////////////////////////////////////////////////
context_t::context_t() : logger(logger::create_logger("tracing", true, true)) {
  this->xptiContextManager = xptiContextManagerGet();

  call_stream_id = xptiRegisterStream(CALL_STREAM_NAME);
  std::ostringstream streamv;
  streamv << STREAM_VER_MAJOR << "." << STREAM_VER_MINOR;
  xptiInitialize(CALL_STREAM_NAME, STREAM_VER_MAJOR, STREAM_VER_MINOR,
                 streamv.str().data());
}

void context_t::notify(uint16_t trace_type, uint32_t id, const char *name,
                       void *args, ur_result_t *resultp, uint64_t instance) {
  xpti::function_with_args_t payload{id, name, args, resultp, nullptr};
  xptiNotifySubscribers(call_stream_id, trace_type, nullptr, activeEvent,
                        instance, &payload);
}

uint64_t context_t::notify_begin(uint32_t id, const char *name, void *args) {
  // we use UINT64_MAX as a special value that means "tracing disabled",
  // so that we don't have to repeat this check in notify_end.
  if (!xptiCheckTraceEnabled(call_stream_id)) {
    return UINT64_MAX;
  }

  if (auto loc = codelocData.get_codeloc()) {
    xpti::payload_t payload =
        xpti::payload_t(loc->functionName, loc->sourceFile, loc->lineNumber,
                        loc->columnNumber, nullptr);
    uint64_t InstanceNumber{};
    activeEvent =
        xptiMakeEvent("Unified Runtime call", &payload, xpti::trace_graph_event,
                      xpti_at::active, &InstanceNumber);
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
