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

extern "C" {

__attribute__((weak)) bool SymbolizeCode(const std::string ModuleName,
                                         uint64_t ModuleOffset,
                                         std::string &Result);
}

namespace ur_sanitizer_layer {

std::string ExtractModuleName(const char *Symbol) {
    auto s1 = std::strrchr(Symbol, '(');
    return std::string(Symbol, s1 - Symbol);
}

// Parse symbolizer output in the following formats:
//   <function_name>
//   <file_name>:<line_number>[:<column_number>]
SourceInfo ParseSymbolizerOutput(std::string Output) {
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

StackTrace GetCurrentBacktrace() {
    void *Frames[MAX_BACKTRACE_FRAMES];
    int FrameCount = backtrace(Frames, MAX_BACKTRACE_FRAMES);
    char **Symbols = backtrace_symbols(Frames, FrameCount);

    if (Symbols == nullptr) {
        return StackTrace();
    }

    StackTrace Stack;
    for (int i = 0; i < FrameCount; i++) {
        if (SymbolizeCode != nullptr) {
            std::string Result;
            std::string ModuleName = ExtractModuleName(Symbols[i]);
            if (SymbolizeCode(ModuleName, (uint64_t)Frames[i], Result)) {
                SourceInfo SrcInfo = ParseSymbolizerOutput(Result);
                std::ostringstream OS;
                if (SrcInfo.file != "??") {
                    OS << "in " << SrcInfo.function << " " << SrcInfo.file
                       << ":" << SrcInfo.line << ":" << SrcInfo.column;
                } else {
                    OS << "in " << SrcInfo.function << " (" << ModuleName << "+"
                       << Frames[i] << ")";
                }
                Stack.stack.emplace_back(OS.str());
                continue;
            }
        }
        Stack.stack.emplace_back(Symbols[i]);
    }
    free(Symbols);

    return Stack;
}

} // namespace ur_sanitizer_layer
