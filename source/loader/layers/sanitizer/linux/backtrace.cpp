/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#include "stacktrace.hpp"

#include <execinfo.h>
#include <string>

namespace ur_sanitizer_layer {

StackTrace GetCurrentBacktrace() {
    void *Frames[MAX_BACKTRACE_FRAMES];
    int FrameCount = backtrace(Frames, MAX_BACKTRACE_FRAMES);
    char **Symbols = backtrace_symbols(Frames, FrameCount);

    if (Symbols == nullptr) {
        return StackTrace();
    }

    StackTrace Stack;
    for (int i = 0; i < FrameCount; i++) {
        BacktraceInfo addr_info(Symbols[i]);
        Stack.stack.emplace_back(addr_info);
    }
    free(Symbols);

    return Stack;
}

} // namespace ur_sanitizer_layer
