/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ur_tracing_layer.cpp
 *
 */
#include "ur_tracing_layer.hpp"
#include "ur_api.h"
#include "ur_util.hpp"
#include "xpti/xpti_data_types.h"
#include "xpti/xpti_trace_framework.h"
#include <sstream>

namespace tracing_layer {
context_t context;

constexpr auto CALL_STREAM_NAME = "ur";
constexpr auto STREAM_VER_MAJOR = UR_MAJOR_VERSION(UR_API_VERSION_CURRENT);
constexpr auto STREAM_VER_MINOR = UR_MINOR_VERSION(UR_API_VERSION_CURRENT);

///////////////////////////////////////////////////////////////////////////////
context_t::context_t() {
    xptiFrameworkInitialize();

    call_stream_id = xptiRegisterStream(CALL_STREAM_NAME);
    std::ostringstream streamv;
    streamv << STREAM_VER_MAJOR << "." << STREAM_VER_MINOR;
    xptiInitialize(CALL_STREAM_NAME, STREAM_VER_MAJOR, STREAM_VER_MINOR,
                   streamv.str().data());
}

bool context_t::isEnabled() { return xptiTraceEnabled(); }

void context_t::notify(uint16_t trace_type, uint32_t id, const char *name,
                       void *args, ur_result_t *resultp, uint64_t instance) {
    xpti::function_with_args_t payload{id, name, args, resultp, nullptr};
    xptiNotifySubscribers(call_stream_id, trace_type, nullptr, nullptr,
                          instance, &payload);
}

uint64_t context_t::notify_begin(uint32_t id, const char *name, void *args) {
    uint64_t instance = xptiGetUniqueId();
    notify((uint16_t)xpti::trace_point_type_t::function_with_args_begin, id,
           name, args, nullptr, instance);
    return instance;
}

void context_t::notify_end(uint32_t id, const char *name, void *args,
                           ur_result_t *resultp, uint64_t instance) {
    notify((uint16_t)xpti::trace_point_type_t::function_with_args_end, id, name,
           args, resultp, instance);
}

///////////////////////////////////////////////////////////////////////////////
context_t::~context_t() {
    xptiFinalize(CALL_STREAM_NAME);

    xptiFrameworkFinalize();
}
} // namespace tracing_layer
