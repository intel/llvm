/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_san_layer.cpp
 *
 */

#include "ur_san_layer.hpp"
#include "asan_interceptor.hpp"
#include "ur_api.h"
#include "ur_util.hpp"

namespace ur_san_layer {
context_t context;

///////////////////////////////////////////////////////////////////////////////
context_t::context_t()
    : interceptor(new SanitizerInterceptor()),
      logger(logger::create_logger("sanitizer")) {}

bool context_t::isAvailable() const { return true; }

///////////////////////////////////////////////////////////////////////////////
context_t::~context_t() {}
} // namespace ur_san_layer
