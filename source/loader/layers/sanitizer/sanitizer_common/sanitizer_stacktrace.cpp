/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer_stacktrace.cpp
 *
 */

#include "sanitizer_stacktrace.hpp"
#include "ur_sanitizer_layer.hpp"

extern "C" {

__attribute__((weak)) void SymbolizeCode(const char *ModuleName,
                                         uint64_t ModuleOffset,
                                         char *ResultString, size_t ResultSize,
                                         size_t *RetSize);
}

namespace ur_sanitizer_layer {

namespace {

bool Contains(const std::string &s, const char *p) {
    return s.find(p) != std::string::npos;
}

// Parse back trace information in the following formats:
//   <module_name>([function_name]+function_offset) [offset]
void ParseBacktraceInfo(const BacktraceInfo &BI, std::string &ModuleName,
                        uptr &Offset) {
    // Parse module name
    size_t End = BI.find_first_of('(');
    assert(End != std::string::npos);
    ModuleName = BI.substr(0, End);
    // Parse offset
    size_t Start = BI.find_first_of('[');
    assert(Start != std::string::npos);
    Start = BI.substr(Start + 1, 2) == "0x" ? Start + 3 : Start + 1;
    End = BI.find_first_of(']');
    assert(End != std::string::npos);
    Offset = std::stoull(BI.substr(Start, End), nullptr, 16);
    return;
}

// Parse symbolizer output in the following formats:
//   <function_name>
//   <file_name>:<line_number>[:<column_number>]
SourceInfo ParseSymbolizerOutput(const std::string &Output) {
    SourceInfo Info;
    // Parse function name
    size_t End = Output.find_first_of('\n');
    assert(End != std::string::npos);
    Info.function = Output.substr(0, End);
    // Parse file name
    size_t Start = End + 1;
    End = Output.find_first_of(':', Start);
    assert(End != std::string::npos);
    Info.file = Output.substr(Start, End - Start);
    // Parse line number
    Start = End + 1;
    End = Output.find_first_of(":\n", Start);
    assert(End != std::string::npos);
    Info.line = std::stoi(Output.substr(Start, End - Start));
    // Parse column number if exists
    if (Output[End] == ':') {
        Start = End + 1;
        End = Output.find_first_of("\n", Start);
        assert(End != std::string::npos);
        Info.column = std::stoi(Output.substr(Start, End - Start));
    }

    return Info;
}

} // namespace

void StackTrace::print() const {
    if (!stack.size()) {
        getContext()->logger.always("  failed to acquire backtrace");
    }

    unsigned index = 0;

    char **BacktraceSymbols = GetBacktraceSymbols(stack);

    for (size_t i = 0; i < stack.size(); i++) {
        BacktraceInfo BI = BacktraceSymbols[i];

        // Skip runtime modules
        if (Contains(BI, "libsycl.so") || Contains(BI, "libur_loader.so") ||
            Contains(BI, "libomptarget.rtl.unified_runtime.so") ||
            Contains(BI, "libomptarget.so")) {
            continue;
        }

        if (&SymbolizeCode != nullptr) {
            std::string Result;
            std::string ModuleName;
            uptr Offset;
            ParseBacktraceInfo(BI, ModuleName, Offset);
            size_t ResultSize = 0;
            SymbolizeCode(ModuleName.c_str(), Offset, nullptr, 0, &ResultSize);
            if (ResultSize) {
                std::vector<char> ResultVector(ResultSize);
                SymbolizeCode(ModuleName.c_str(), Offset, ResultVector.data(),
                              ResultSize, nullptr);
                std::string Result((char *)ResultVector.data());
                SourceInfo SrcInfo = ParseSymbolizerOutput(Result);
                if (SrcInfo.file != "??") {
                    getContext()->logger.always(" #{} in {} {}:{}:{}", index,
                                                SrcInfo.function, SrcInfo.file,
                                                SrcInfo.line, SrcInfo.column);
                } else {
                    getContext()->logger.always(" #{} in {} ({}+{})", index,
                                                SrcInfo.function, ModuleName,
                                                (void *)Offset);
                }
            }
        } else {
            getContext()->logger.always("  #{} {}", index, BI);
        }
        ++index;
    }
    getContext()->logger.always("");

    free(BacktraceSymbols);
}

} // namespace ur_sanitizer_layer
