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
#include "ur_sanitizer_layer.hpp"

#include <cstring>
#include <execinfo.h>
#include <string>

namespace ur_sanitizer_layer {

bool ExtractSymbolInfo(const char *symbol, AddressInfo &info) {
    info.debug = std::string(symbol);

    auto s1 = std::strrchr(symbol, '(');
    info.module = std::string(symbol, s1 - symbol);
    auto s2 = std::strrchr(symbol, '[');
    info.offset = std::stoull(s2 + 1, nullptr, 16);
    return true;
}

StackTrace GetCurrentBacktrace() {
    void *backtraceFrames[MAX_BACKTRACE_FRAMES];
    int frameCount = backtrace(backtraceFrames, MAX_BACKTRACE_FRAMES);
    char **backtraceStr = backtrace_symbols(backtraceFrames, frameCount);

    if (backtraceStr == nullptr) {
        return StackTrace();
    }

    StackTrace stack;
    for (int i = 0; i < frameCount; i++) {
        AddressInfo addr_info;
        if (!ExtractSymbolInfo(backtraceStr[i], addr_info)) {
            continue;
        }
        stack.stack.emplace_back(std::move(addr_info));
    }
    free(backtraceStr);

    return stack;
}

} // namespace ur_sanitizer_layer
