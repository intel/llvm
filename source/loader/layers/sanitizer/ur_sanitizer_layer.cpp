/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_sanitizer_layer.cpp
 *
 */

#include "ur_sanitizer_layer.hpp"
#include "asan_interceptor.hpp"

namespace ur_sanitizer_layer {
context_t context;

///////////////////////////////////////////////////////////////////////////////
context_t::context_t()
    : interceptor(std::make_unique<SanitizerInterceptor>()),
      logger(logger::create_logger("sanitizer")) {}

bool context_t::isAvailable() const { return true; }

ur_result_t context_t::tearDown() { return UR_RESULT_SUCCESS; }

///////////////////////////////////////////////////////////////////////////////
context_t::~context_t() {}
} // namespace ur_sanitizer_layer
