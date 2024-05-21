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

constexpr size_t SHADOW_SIZE = 0x80000000000ULL;
uptr SHADOW_BEGIN;
uptr SHADOW_END;

bool IsShadowMemInited = false;

ur_result_t SetupShadowMemory(uptr &ShadowBegin, uptr &ShadowEnd) {
    static ur_result_t Result = []() {
        SHADOW_BEGIN = MmapNoReserve(0, SHADOW_SIZE);
        if (SHADOW_BEGIN == 0) {
            return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }
        SHADOW_END = SHADOW_BEGIN + SHADOW_SIZE;
        IsShadowMemInited = true;
        return UR_RESULT_SUCCESS;
    }();
    ShadowBegin = SHADOW_BEGIN;
    ShadowEnd = SHADOW_END;
    return Result;
}

ur_result_t DestroyShadowMemory() {
    if (!IsShadowMemInited) {
        return UR_RESULT_SUCCESS;
    }
    if (!Munmap(SHADOW_BEGIN, SHADOW_SIZE)) {
        return UR_RESULT_ERROR_UNKNOWN;
    }
    return UR_RESULT_SUCCESS;
}

} // namespace cpu

namespace pvc {

///   Host/Shared USM : 0x0              ~ 0x07ff_ffff_ffff
///   Device USM      : 0x0800_0000_0000 ~ 0x17ff_ffff_ffff
constexpr size_t SHADOW_SIZE = 0x180000000000ULL;

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
