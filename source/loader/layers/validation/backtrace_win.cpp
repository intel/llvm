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

#include <Windows.h>
// Windows.h must be included before DbgHelp.h
#include <DbgHelp.h>
#include <vector>

namespace ur_validation_layer {

std::vector<BacktraceLine> getCurrentBacktrace() {
    HANDLE process = GetCurrentProcess();
    SymInitialize(process, nullptr, true);

    PVOID frames[MAX_BACKTRACE_FRAMES];
    WORD frameCount =
        CaptureStackBackTrace(0, MAX_BACKTRACE_FRAMES, frames, NULL);

    if (frameCount == 0) {
        SymCleanup(process);
        return std::vector<BacktraceLine>(1, "Failed to acquire a backtrace");
    }

    DWORD displacement = 0;
    IMAGEHLP_LINE64 line;
    line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);

    std::vector<BacktraceLine> backtrace;
    try {
        for (int i = 0; i < frameCount; i++) {
            if (SymGetLineFromAddr64(process, (DWORD64)frames[i], &displacement,
                                     &line)) {
                backtrace.push_back(std::string(line.FileName) + ":" +
                                    std::to_string(line.LineNumber));
            } else {
                backtrace.push_back("????????");
            }
        }
    } catch (std::bad_alloc &) {
        SymCleanup(process);
        return std::vector<BacktraceLine>(1, "Failed to acquire a backtrace");
    }

    SymCleanup(process);

    return backtrace;
}

} // namespace ur_validation_layer
