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

#include "asan_report.hpp"
#include "asan_allocator.hpp"
#include "asan_interceptor.hpp"
#include "device_sanitizer_report.hpp"
#include "ur_sanitizer_layer.hpp"
#include "ur_sanitizer_utils.hpp"

namespace ur_sanitizer_layer {

void ReportBadFree(uptr Addr, const StackTrace &stack,
                   const std::shared_ptr<AllocInfo> &AI) {
    context.logger.always(
        "\n====ERROR: DeviceSanitizer: bad-free on address {}", (void *)Addr);
    stack.Print();

    if (!AI) {
        context.logger.always("{} may be allocated on Host Memory",
                              (void *)Addr);
        exit(1);
    }

    assert(!AI->IsReleased && "Chunk must be not released");

    context.logger.always("{} is located inside of {} region [{}, {})",
                          (void *)Addr, ToString(AI->Type),
                          (void *)AI->UserBegin, (void *)(AI->UserEnd + 1));
    context.logger.always("allocated here:");
    AI->AllocStack.Print();

    exit(1);
}

void ReportBadContext(uptr Addr, const StackTrace &stack,
                      const std::shared_ptr<AllocInfo> &AI) {
    context.logger.always(
        "\n====ERROR: DeviceSanitizer: bad-context on address {}",
        (void *)Addr);
    stack.Print();

    context.logger.always("{} is located inside of {} region [{}, {})",
                          (void *)Addr, ToString(AI->Type),
                          (void *)AI->UserBegin, (void *)(AI->UserEnd + 1));
    context.logger.always("allocated here:");
    AI->AllocStack.Print();

    if (AI->IsReleased) {
        context.logger.always("freed here:");
        AI->ReleaseStack.Print();
    }

    exit(1);
}

void ReportDoubleFree(uptr Addr, const StackTrace &Stack,
                      const std::shared_ptr<AllocInfo> &AI) {
    context.logger.always(
        "\n====ERROR: DeviceSanitizer: double-free on address {}",
        (void *)Addr);
    Stack.Print();

    context.logger.always("{} is located inside of {} region [{}, {})",
                          (void *)Addr, ToString(AI->Type),
                          (void *)AI->UserBegin, (void *)(AI->UserEnd + 1));
    context.logger.always("freed here:");
    AI->ReleaseStack.Print();
    context.logger.always("previously allocated here:");
    AI->AllocStack.Print();
    exit(1);
}

void ReportGenericError(const DeviceSanitizerReport &Report) {
    context.logger.always("\n====ERROR: DeviceSanitizer: {}",
                          ToString(Report.ErrorType));
    exit(1);
}

void ReportOutOfBoundsError(const DeviceSanitizerReport &Report,
                            ur_kernel_handle_t Kernel) {
    const char *File = Report.File[0] ? Report.File : "<unknown file>";
    const char *Func = Report.Func[0] ? Report.Func : "<unknown func>";
    auto KernelName = GetKernelName(Kernel);

    // Try to demangle the kernel name
    KernelName = DemangleName(KernelName);

    context.logger.always("\n====ERROR: DeviceSanitizer: {} on {}",
                          ToString(Report.ErrorType),
                          ToString(Report.MemoryType));
    context.logger.always(
        "{} of size {} at kernel <{}> LID({}, {}, {}) GID({}, "
        "{}, {})",
        Report.IsWrite ? "WRITE" : "READ", Report.AccessSize,
        KernelName.c_str(), Report.LID0, Report.LID1, Report.LID2, Report.GID0,
        Report.GID1, Report.GID2);
    context.logger.always("  #0 {} {}:{}", Func, File, Report.Line);

    exit(1);
}

void ReportUseAfterFree(const DeviceSanitizerReport &Report,
                        ur_kernel_handle_t Kernel,
                        ur_context_handle_t Context) {
    const char *File = Report.File[0] ? Report.File : "<unknown file>";
    const char *Func = Report.Func[0] ? Report.Func : "<unknown func>";
    auto KernelName = GetKernelName(Kernel);

    // Try to demangle the kernel name
    KernelName = DemangleName(KernelName);

    context.logger.always("\n====ERROR: DeviceSanitizer: {} on address {}",
                          ToString(Report.ErrorType), (void *)Report.Addr);
    context.logger.always(
        "{} of size {} at kernel <{}> LID({}, {}, {}) GID({}, "
        "{}, {})",
        Report.IsWrite ? "WRITE" : "READ", Report.AccessSize,
        KernelName.c_str(), Report.LID0, Report.LID1, Report.LID2, Report.GID0,
        Report.GID1, Report.GID2);
    context.logger.always("  #0 {} {}:{}", Func, File, Report.Line);
    context.logger.always("");

    auto AllocInfoItOp =
        context.interceptor->findAllocInfoByAddress(Report.Addr);
    if (!AllocInfoItOp) {
        context.logger.always("Failed to find which chunck {} is allocated",
                              (void *)Report.Addr);
        return;
    }

    auto &AllocInfo = (*AllocInfoItOp)->second;
    if (AllocInfo->Context != Context) {
        context.logger.always("Failed to find which chunck {} is allocated",
                              (void *)Report.Addr);
    }
    assert(AllocInfo->IsReleased);

    context.logger.always("{} is located inside of {} region [{}, {})",
                          (void *)Report.Addr, ToString(AllocInfo->Type),
                          (void *)AllocInfo->UserBegin,
                          (void *)(AllocInfo->UserEnd + 1));
    context.logger.always("allocated here:");
    AllocInfo->AllocStack.Print();
    context.logger.always("released here:");
    AllocInfo->ReleaseStack.Print();

    exit(1);
}

} // namespace ur_sanitizer_layer
