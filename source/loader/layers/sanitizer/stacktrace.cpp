/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file stacktrace.cpp
 *
 */

#pragma once

#include "stacktrace.hpp"
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {

#define MAX_BACKTRACE_FRAMES 64

void StackTrace::Print() const {
    if (!stack.size()) {
        context.logger.always("  failed to acquire backtrace");
    }
    for (unsigned i = 0; i < stack.size(); ++i) {
        context.logger.always("  #{} {}", i, stack[i]);
    }
    context.logger.always("");
}

} // namespace ur_sanitizer_layer