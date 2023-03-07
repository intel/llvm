/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ur_lib.cpp
 *
 */
#include "ur_lib.hpp"
#include "logger/ur_logger.hpp"
#include "ur_loader.hpp"
#include "ur_proxy_layer.hpp"
#include "validation/ur_validation_layer.hpp"

#if UR_ENABLE_TRACING
#include "tracing/ur_tracing_layer.hpp"
#endif

namespace logger {
Logger logger = create_logger("loader");
}

namespace ur_lib {
///////////////////////////////////////////////////////////////////////////////
context_t *context;

///////////////////////////////////////////////////////////////////////////////
context_t::context_t() {}

///////////////////////////////////////////////////////////////////////////////
context_t::~context_t() {}

//////////////////////////////////////////////////////////////////////////
__urdlllocal ur_result_t context_t::Init(ur_device_init_flags_t device_flags) {
    ur_result_t result;
    result = loader::context->init();

    if (UR_RESULT_SUCCESS == result) {
        result = urInit();
    }

    proxy_layer_context_t *layers[] = {
        &validation_layer::context,
#if UR_ENABLE_TRACING
        &tracing_layer::context
#endif
    };

    for (proxy_layer_context_t *l : layers) {
        if (l->isEnabled()) {
            l->init(&context->urDdiTable);
        }
    }

    return result;
}

} // namespace ur_lib
