/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file backtrace.cpp
 *
 */
#include "sanitizer_common/sanitizer_stacktrace.hpp"

#include <execinfo.h>
#include <string>

namespace ur_sanitizer_layer {

StackTrace GetCurrentBacktrace() {
    BacktraceFrame Frames[MAX_BACKTRACE_FRAMES];
    int FrameCount = backtrace(Frames, MAX_BACKTRACE_FRAMES);

    StackTrace Stack;
    Stack.stack =
        std::vector<BacktraceFrame>(&Frames[0], &Frames[FrameCount - 1]);

    return Stack;
}

char **GetBacktraceSymbols(const std::vector<BacktraceFrame> &BacktraceFrames) {
    assert(!BacktraceFrames.empty());

    char **BacktraceSymbols =
        backtrace_symbols(&BacktraceFrames[0], BacktraceFrames.size());
    return BacktraceSymbols;
}

} // namespace ur_sanitizer_layer
