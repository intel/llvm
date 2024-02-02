/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#include "backtrace.hpp"

#include <execinfo.h>
#include <vector>

namespace ur_sanitizer_layer {

std::vector<BacktraceLine> getCurrentBacktrace() {
    void *backtraceFrames[MAX_BACKTRACE_FRAMES];
    int frameCount = backtrace(backtraceFrames, MAX_BACKTRACE_FRAMES);
    char **backtraceStr = backtrace_symbols(backtraceFrames, frameCount);

    if (backtraceStr == nullptr) {
        return std::vector<BacktraceLine>(1, "Failed to acquire a backtrace");
    }

    std::vector<BacktraceLine> backtrace;
    try {
        for (int i = 0; i < frameCount; i++) {
            backtrace.emplace_back(backtraceStr[i]);
        }
    } catch (std::bad_alloc &) {
        free(backtraceStr);
        return std::vector<BacktraceLine>(1, "Failed to acquire a backtrace");
    }

    free(backtraceStr);

    return backtrace;
}

} // namespace ur_sanitizer_layer
