/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file stacktrace.cpp
 *
 */

#include "stacktrace.hpp"
#include "symbolizer_llvm.hpp"
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {

namespace {

std::string GetFileName(const std::string &FilePath) {
    auto p = FilePath.find_last_of('/');
    return FilePath.substr(p + 1);
}

bool StartWith(const std::string &Str, const char *Pattern) {
    return Str.rfind(Pattern, 0) == 0;
}

} // namespace

void StackTrace::Print() const {
    if (!stack.size()) {
        context.logger.always("  failed to acquire backtrace");
    }

    unsigned index = 0;

    for (auto &BI : stack) {
        auto ModuleFile = GetFileName(BI.module);
        if (StartWith(ModuleFile, "libsycl.so") ||
            StartWith(ModuleFile, "libpi_unified_runtime.so") ||
            StartWith(ModuleFile, "libur_loader.so")) {
            continue;
        }

        SourceInfo SI;
        for (auto &symbolizer : SymbolizerTools) {
            if (symbolizer->SymbolizePC(BI, SI)) {
                break;
            }
        }

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
