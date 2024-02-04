/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#include "stacktrace.hpp"

#include <execinfo.h>

namespace ur_sanitizer_layer {

StackTrace GetCurrentBacktrace() {
    void *backtraceFrames[MAX_BACKTRACE_FRAMES];
    int frameCount = backtrace(backtraceFrames, MAX_BACKTRACE_FRAMES);
    char **backtraceStr = backtrace_symbols(backtraceFrames, frameCount);

    if (backtraceStr == nullptr) {
        return StackTrace();
    }

    StackTrace stack;
    for (int i = 0; i < frameCount; i++) {
        stack.stack.emplace_back(backtraceStr[i]);
    }
    free(backtraceStr);

    return stack;
}

} // namespace ur_sanitizer_layer
