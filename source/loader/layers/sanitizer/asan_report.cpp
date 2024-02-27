/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_report.cpp
 *
 */

#pragma once

#include "asan_report.hpp"
#include "asan_allocator.hpp"
#include "device_sanitizer_report.hpp"
#include "ur_sanitizer_layer.hpp"
#include "ur_sanitizer_utils.hpp"

namespace ur_sanitizer_layer {

void ReportBadFree(uptr Addr, const StackTrace &stack,
                   std::shared_ptr<USMAllocInfo> AllocInfo) {
    context.logger.always(
        "\n====ERROR: DeviceSanitizer: attempting free on address which "
        "was not malloc()-ed: {} in thread T0",
        (void *)Addr);
    stack.Print();

    if (!AllocInfo) { // maybe Addr is host allocated memory
        context.logger.always("{} is maybe allocated on Host Memory",
                              (void *)Addr);
        exit(1);
    }

    assert(!AllocInfo->IsReleased && "Chunk must be not released");

    context.logger.always("{} is located inside of {} region [{}, {}]",
                          (void *)Addr, getFormatString(AllocInfo->Type),
                          (void *)AllocInfo->UserBegin,
                          (void *)AllocInfo->UserEnd);
    context.logger.always("allocated by thread T0 here:");
    AllocInfo->AllocStack.Print();

    exit(1);
}

void ReportDoubleFree(uptr Addr, const StackTrace &Stack,
                      std::shared_ptr<USMAllocInfo> AllocInfo) {
    context.logger.always("\n====ERROR: DeviceSanitizer: double-free on {}",
                          (void *)Addr);
    Stack.Print();
    AllocInfo->AllocStack.Print();
    AllocInfo->ReleaseStack.Print();
    exit(1);
}

void ReportGenericError(DeviceSanitizerReport &Report,
                        ur_kernel_handle_t Kernel, ur_context_handle_t Context,
                        ur_device_handle_t Device) {
    const char *File = Report.File[0] ? Report.File : "<unknown file>";
    const char *Func = Report.Func[0] ? Report.Func : "<unknown func>";
    auto KernelName = getKernelName(Kernel);

    context.logger.always("\n====ERROR: DeviceSanitizer: {} on {}",
                          DeviceSanitizerFormat(Report.ErrorType),
                          DeviceSanitizerFormat(Report.MemoryType));
    context.logger.always(
        "{} of size {} at kernel <{}> LID({}, {}, {}) GID({}, "
        "{}, {})",
        Report.IsWrite ? "WRITE" : "READ", Report.AccessSize,
        KernelName.c_str(), Report.LID0, Report.LID1, Report.LID2, Report.GID0,
        Report.GID1, Report.GID2);
    context.logger.always("  #0 {} {}:{}\n", Func, File, Report.Line);

    if (Report.ErrorType == DeviceSanitizerErrorType::USE_AFTER_FREE) {
        auto AllocInfos =
            context.interceptor->findAllocInfoByAddress(Report.Addr);
        if (!AllocInfos.size()) {
            context.logger.always("can't find which chunck {} is allocated",
                                  (void *)Report.Addr);
        }
        for (auto &AllocInfo : AllocInfos) {
            if (!AllocInfo->IsReleased) {
                continue;
            }
            context.logger.always(
                "{} is located inside of {} region [{}, {}]",
                (void *)Report.Addr, getFormatString(AllocInfo->Type),
                (void *)AllocInfo->UserBegin, (void *)AllocInfo->UserEnd);
            context.logger.always("allocated by thread T0 here:");
            AllocInfo->AllocStack.Print();
            context.logger.always("released by thread T0 here:");
            AllocInfo->ReleaseStack.Print();
        }
    }

    exit(1);
}

} // namespace ur_sanitizer_layer
