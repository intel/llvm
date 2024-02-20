/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_shadow_setup.cpp
 *
 */

#include "asan_shadow_setup.hpp"
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {

namespace cpu {

// Based on "compiler-rt/lib/asan/asan_mapping.h"
// Typical shadow mapping on Linux/x86_64 with SHADOW_OFFSET == 0x00007fff8000:
constexpr uptr LOW_SHADOW_BEGIN = 0x00007fff8000ULL;
constexpr uptr LOW_SHADOW_END = 0x00008fff6fffULL;
constexpr uptr SHADOW_GAP_BEGIN = 0x00008fff7000ULL;
constexpr uptr SHADOW_GAP_END = 0x02008fff6fffULL;
constexpr uptr HIGH_SHADOW_BEGIN = 0x02008fff7000ULL;
constexpr uptr HIGH_SHADOW_END = 0x10007fff7fffULL;
constexpr uptr LOW_SHADOW_SIZE = LOW_SHADOW_END - LOW_SHADOW_BEGIN;
constexpr uptr SHADOW_GAP_SIZE = SHADOW_GAP_END - SHADOW_GAP_BEGIN;
constexpr uptr HIGH_SHADOW_SIZE = HIGH_SHADOW_END - HIGH_SHADOW_BEGIN;

bool IsShadowMemInited;

ur_result_t SetupShadowMemory(uptr &ShadowBegin, uptr &ShadowEnd) {
    static ur_result_t Result = []() {
        if (!MmapFixedNoReserve(LOW_SHADOW_BEGIN, LOW_SHADOW_SIZE)) {
            return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }
        if (!MmapFixedNoReserve(HIGH_SHADOW_BEGIN, HIGH_SHADOW_SIZE)) {
            return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }
        if (!MmapFixedNoAccess(SHADOW_GAP_BEGIN, SHADOW_GAP_SIZE)) {
            return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }
        IsShadowMemInited = true;
        return UR_RESULT_SUCCESS;
    }();
    ShadowBegin = LOW_SHADOW_BEGIN;
    ShadowEnd = HIGH_SHADOW_END;
    return Result;
}

ur_result_t DestroyShadowMemory() {
    static ur_result_t Result = []() {
        if (!IsShadowMemInited) {
            return UR_RESULT_SUCCESS;
        }
        if (!Munmap(LOW_SHADOW_BEGIN, LOW_SHADOW_SIZE)) {
            return UR_RESULT_ERROR_UNKNOWN;
        }
        if (!Munmap(HIGH_SHADOW_BEGIN, HIGH_SHADOW_SIZE)) {
            return UR_RESULT_ERROR_UNKNOWN;
        }
        if (!Munmap(SHADOW_GAP_BEGIN, SHADOW_GAP_SIZE)) {
            return UR_RESULT_ERROR_UNKNOWN;
        }
        return UR_RESULT_SUCCESS;
    }();
    return Result;
}

} // namespace cpu

namespace pvc {

/// SHADOW MEMORY MAPPING (PVC, with CPU 47bit)
///   Host/Shared USM : 0x0              ~ 0x0fff_ffff_ffff
///   ?               : 0x1000_0000_0000 ~ 0x1fff_ffff_ffff
///   Device USM      : 0x2000_0000_0000 ~ 0x3fff_ffff_ffff
constexpr size_t SHADOW_SIZE = 1ULL << 46;

uptr LOW_SHADOW_BEGIN;
uptr HIGH_SHADOW_END;

ur_context_handle_t ShadowContext;

ur_result_t SetupShadowMemory(ur_context_handle_t Context, uptr &ShadowBegin,
                              uptr &ShadowEnd) {
    // Currently, Level-Zero doesn't create independent VAs for each contexts, if we reserve
    // shadow memory for each contexts, this will cause out-of-resource error when user uses
    // multiple contexts. Therefore, we just create one shadow memory here.
    static ur_result_t Result = [&Context]() {
        // TODO: Protect Bad Zone
        auto Result = context.urDdiTable.VirtualMem.pfnReserve(
            Context, nullptr, SHADOW_SIZE, (void **)&LOW_SHADOW_BEGIN);
        if (Result == UR_RESULT_SUCCESS) {
            HIGH_SHADOW_END = LOW_SHADOW_BEGIN + SHADOW_SIZE;
            // Retain the context which reserves shadow memory
            ShadowContext = Context;
            context.urDdiTable.Context.pfnRetain(Context);
        }
        return Result;
    }();
    ShadowBegin = LOW_SHADOW_BEGIN;
    ShadowEnd = HIGH_SHADOW_END;
    return Result;
}

ur_result_t DestroyShadowMemory() {
    static ur_result_t Result = []() {
        if (!ShadowContext) {
            return UR_RESULT_SUCCESS;
        }
        auto Result = context.urDdiTable.VirtualMem.pfnFree(
            ShadowContext, (const void *)LOW_SHADOW_BEGIN, SHADOW_SIZE);
        context.urDdiTable.Context.pfnRelease(ShadowContext);
        return Result;
    }();
    return Result;
}

} // namespace pvc

ur_result_t SetupShadowMemoryOnCPU(uptr &ShadowBegin, uptr &ShadowEnd) {
    return cpu::SetupShadowMemory(ShadowBegin, ShadowEnd);
}

ur_result_t DestroyShadowMemoryOnCPU() { return cpu::DestroyShadowMemory(); }

ur_result_t SetupShadowMemoryOnPVC(ur_context_handle_t Context,
                                   uptr &ShadowBegin, uptr &ShadowEnd) {
    return pvc::SetupShadowMemory(Context, ShadowBegin, ShadowEnd);
}

ur_result_t DestroyShadowMemoryOnPVC() { return pvc::DestroyShadowMemory(); }

} // namespace ur_sanitizer_layer
