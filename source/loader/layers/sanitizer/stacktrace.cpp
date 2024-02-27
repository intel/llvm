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
#include "symbolizer_llvm.hpp"
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {

#define MAX_BACKTRACE_FRAMES 64

std::string getFileName(const std::string &FilePath) {
    auto p = FilePath.find_last_of('/');
    return FilePath.substr(p + 1);
}

bool startWith(const std::string &Str, const char *Pattern) {
    return Str.rfind(Pattern, 0) == 0;
}

void StackTrace::Print() const {
    if (!stack.size()) {
        context.logger.always("  failed to acquire backtrace");
    }

    LLVMSymbolizer symbolizer;
    unsigned index = 0;

    for (auto &BI : stack) {
        auto ModuleFile = getFileName(BI.module);
        if (startWith(ModuleFile, "libsycl.so") ||
            startWith(ModuleFile, "libpi_unified_runtime.so") ||
            startWith(ModuleFile, "libur_loader.so")) {
            continue;
        }

        SourceInfo SI;
        symbolizer.SymbolizePC(BI, SI);

        if (!SI.file.empty()) {
            context.logger.always("  #{} {} in {} {}:{}:{}", index,
                                  (void *)BI.offset, SI.function, SI.file,
                                  SI.line, SI.column);
        } else {
            context.logger.always("  #{} {} in {} {}", index, (void *)BI.offset,
                                  SI.function, BI.module);
        }
        ++index;
    }
    context.logger.always("");
}

} // namespace ur_sanitizer_layer