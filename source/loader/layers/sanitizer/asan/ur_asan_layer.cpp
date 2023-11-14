/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_asan_layer.cpp
 *
 */
#include "ur_asan_layer.hpp"
#include "sanitizer_interceptor.hpp"
#include "ur_api.h"
#include "ur_util.hpp"

#include <sstream>

namespace ur_asan_layer {
context_t context;

///////////////////////////////////////////////////////////////////////////////
context_t::context_t()
    : interceptor(new SanitizerInterceptor(urDdiTable)),
      logger(logger::create_logger("asan")) {}

bool context_t::isAvailable() const { return true; }

///////////////////////////////////////////////////////////////////////////////
context_t::~context_t() {}
} // namespace ur_asan_layer
