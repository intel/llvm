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
#include "asan_libdevice.hpp"
#include "asan_options.hpp"
#include "asan_validator.hpp"
#include "sanitizer_common/sanitizer_utils.hpp"
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {
namespace asan {

namespace {

void PrintAllocateInfo(uptr Addr, const AllocInfo *AI) {
    getContext()->logger.always("{} is located inside of {} region [{}, {})",
                                (void *)Addr, ToString(AI->Type),
                                (void *)AI->UserBegin, (void *)AI->UserEnd);
    getContext()->logger.always("allocated here:");
    AI->AllocStack.print();
    if (AI->IsReleased) {
        getContext()->logger.always("freed here:");
        AI->ReleaseStack.print();
    }
}

} // namespace

void ReportBadFree(uptr Addr, const StackTrace &stack,
                   const std::shared_ptr<AllocInfo> &AI) {
    getContext()->logger.always(
        "\n====ERROR: DeviceSanitizer: bad-free on address {}", (void *)Addr);
    stack.print();

    if (!AI) {
        getContext()->logger.always("{} may be allocated on Host Memory",
                                    (void *)Addr);
    } else {
        assert(!AI->IsReleased && "Chunk must be not released");
        PrintAllocateInfo(Addr, AI.get());
    }
}

void ReportBadContext(uptr Addr, const StackTrace &stack,
                      const std::shared_ptr<AllocInfo> &AI) {
    getContext()->logger.always(
        "\n====ERROR: DeviceSanitizer: bad-context on address {}",
        (void *)Addr);
    stack.print();

    PrintAllocateInfo(Addr, AI.get());
}

void ReportDoubleFree(uptr Addr, const StackTrace &Stack,
                      const std::shared_ptr<AllocInfo> &AI) {
    getContext()->logger.always(
        "\n====ERROR: DeviceSanitizer: double-free on address {}",
        (void *)Addr);
    Stack.print();

    getContext()->logger.always("{} is located inside of {} region [{}, {})",
                                (void *)Addr, ToString(AI->Type),
                                (void *)AI->UserBegin, (void *)AI->UserEnd);
    getContext()->logger.always("freed here:");
    AI->ReleaseStack.print();
    getContext()->logger.always("previously allocated here:");
    AI->AllocStack.print();
}

void ReportMemoryLeak(const std::shared_ptr<AllocInfo> &AI) {
    getContext()->logger.always(
        "\n====ERROR: DeviceSanitizer: detected memory leaks of {}",
        ToString(AI->Type));
    getContext()->logger.always(
        "Direct leak of {} byte(s) at {} allocated from:",
        AI->UserEnd - AI->UserBegin, (void *)AI->UserBegin);
    AI->AllocStack.print();
}

void ReportFatalError(const AsanErrorReport &Report) {
    getContext()->logger.always("\n====ERROR: DeviceSanitizer: {}",
                                ToString(Report.ErrorTy));
}

void ReportGenericError(const AsanErrorReport &Report,
                        ur_kernel_handle_t Kernel) {
    const char *File = Report.File[0] ? Report.File : "<unknown file>";
    const char *Func = Report.Func[0] ? Report.Func : "<unknown func>";
    auto KernelName = GetKernelName(Kernel);

    // Try to demangle the kernel name
    KernelName = DemangleName(KernelName);

    getContext()->logger.always(
        "\n====ERROR: DeviceSanitizer: {} on {} ({})", ToString(Report.ErrorTy),
        ToString(Report.MemoryTy), (void *)Report.Address);
    getContext()->logger.always(
        "{} of size {} at kernel <{}> LID({}, {}, {}) GID({}, "
        "{}, {})",
        Report.IsWrite ? "WRITE" : "READ", Report.AccessSize,
        KernelName.c_str(), Report.LID0, Report.LID1, Report.LID2, Report.GID0,
        Report.GID1, Report.GID2);
    getContext()->logger.always("  #0 {} {}:{}", Func, File, Report.Line);
}

void ReportUseAfterFree(const AsanErrorReport &Report,
                        ur_kernel_handle_t Kernel,
                        ur_context_handle_t Context) {
    const char *File = Report.File[0] ? Report.File : "<unknown file>";
    const char *Func = Report.Func[0] ? Report.Func : "<unknown func>";
    auto KernelName = GetKernelName(Kernel);

    // Try to demangle the kernel name
    KernelName = DemangleName(KernelName);

    getContext()->logger.always(
        "\n====ERROR: DeviceSanitizer: {} on address {}",
        ToString(Report.ErrorTy), (void *)Report.Address);
    getContext()->logger.always(
        "{} of size {} at kernel <{}> LID({}, {}, {}) GID({}, "
        "{}, {})",
        Report.IsWrite ? "WRITE" : "READ", Report.AccessSize,
        KernelName.c_str(), Report.LID0, Report.LID1, Report.LID2, Report.GID0,
        Report.GID1, Report.GID2);
    getContext()->logger.always("  #0 {} {}:{}", Func, File, Report.Line);
    getContext()->logger.always("");

    if (getAsanInterceptor()->getOptions().MaxQuarantineSizeMB > 0) {
        auto AllocInfoItOp =
            getAsanInterceptor()->findAllocInfoByAddress(Report.Address);

        if (!AllocInfoItOp) {
            getContext()->logger.always(
                "Failed to find which chunck {} is allocated",
                (void *)Report.Address);
        } else {
            auto &AllocInfo = (*AllocInfoItOp)->second;
            if (AllocInfo->Context != Context) {
                getContext()->logger.always(
                    "Failed to find which chunck {} is allocated",
                    (void *)Report.Address);
            }
            assert(AllocInfo->IsReleased &&
                   "It must be released since it's use-after-free");

            PrintAllocateInfo(Report.Address, AllocInfo.get());
        }
    } else {
        getContext()->logger.always(
            "Please enable quarantine to get more information like memory "
            "chunck's kind and where the chunck was allocated and released.");
    }
}

void ReportInvalidKernelArgument(ur_kernel_handle_t Kernel, uint32_t ArgIndex,
                                 uptr Addr, const ValidateUSMResult &VR,
                                 StackTrace Stack) {
    getContext()->logger.always("\n====ERROR: DeviceSanitizer: "
                                "invalid-argument on kernel <{}>",
                                DemangleName(GetKernelName(Kernel)));
    Stack.print();
    auto &AI = VR.AI;
    ArgIndex = ArgIndex + 1;
    switch (VR.Type) {
    case ValidateUSMResult::MAYBE_HOST_POINTER:
        getContext()->logger.always("The {}th argument {} is not a USM pointer",
                                    ArgIndex, (void *)Addr);
        break;
    case ValidateUSMResult::RELEASED_POINTER:
        getContext()->logger.always(
            "The {}th argument {} is a released USM pointer", ArgIndex + 1,
            (void *)Addr);
        PrintAllocateInfo(Addr, AI.get());
        break;
    case ValidateUSMResult::BAD_CONTEXT:
        getContext()->logger.always(
            "The {}th argument {} is allocated in other context", ArgIndex + 1,
            (void *)Addr);
        PrintAllocateInfo(Addr, AI.get());
        break;
    case ValidateUSMResult::BAD_DEVICE:
        getContext()->logger.always(
            "The {}th argument {} is allocated in other device", ArgIndex + 1,
            (void *)Addr);
        PrintAllocateInfo(Addr, AI.get());
        break;
    case ValidateUSMResult::OUT_OF_BOUNDS:
        getContext()->logger.always(
            "The {}th argument {} is located outside of its region [{}, {})",
            ArgIndex + 1, (void *)Addr, (void *)AI->UserBegin,
            (void *)AI->UserEnd);
        getContext()->logger.always("allocated here:");
        AI->AllocStack.print();
        break;
    default:
        break;
    }
}

} // namespace asan
} // namespace ur_sanitizer_layer
