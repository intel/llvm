/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_sanitizer_layer.cpp
 *
 */

#include "ur_sanitizer_layer.hpp"
#include "asan/asan_ddi.hpp"
#include "msan/msan_ddi.hpp"

namespace ur_sanitizer_layer {
context_t *getContext() { return context_t::get_direct(); }

///////////////////////////////////////////////////////////////////////////////
context_t::context_t()
    : logger(logger::create_logger("sanitizer", false, false,
                                   logger::Level::WARN)) {}

ur_result_t context_t::tearDown() {
    switch (enabledType) {
    case SanitizerType::AddressSanitizer:
        destroyAsanInterceptor();
        break;
    case SanitizerType::MemorySanitizer:
        destroyMsanInterceptor();
        break;
    default:
        break;
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
context_t::~context_t() {}
} // namespace ur_sanitizer_layer
