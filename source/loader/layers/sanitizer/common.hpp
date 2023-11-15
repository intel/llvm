/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file common.hpp
 *
 */

#pragma once

#include "ur/ur.hpp"
#include "ur_ddi.h"

#include <cassert>
#include <cstdint>

namespace ur_sanitizer_layer {

// ================================================================
// Copy from LLVM compiler-rt/lib/asan

typedef uintptr_t uptr;
typedef unsigned char u8;
typedef unsigned int u32;

#define ASAN_SHADOW_SCALE 3
#define ASAN_SHADOW_GRANULARITY (1ULL << ASAN_SHADOW_SCALE)

inline constexpr bool IsPowerOfTwo(uptr x) {
    return (x & (x - 1)) == 0 && x != 0;
}

inline constexpr uptr RoundUpTo(uptr Size, uptr boundary) {
    assert(IsPowerOfTwo(boundary));
    return (Size + boundary - 1) & ~(boundary - 1);
}

inline constexpr uptr RoundDownTo(uptr x, uptr boundary) {
    assert(IsPowerOfTwo(boundary));
    return x & ~(boundary - 1);
}

inline constexpr bool IsAligned(uptr a, uptr alignment) {
    return (a & (alignment - 1)) == 0;
}

// Valid redzone sizes are 16, 32, 64, ... 2048, so we encode them in 3 bits.
// We use adaptive redzones: for larger allocation larger redzones are used.
inline constexpr u32 RZLog2Size(u32 rz_log) {
    // CHECK_LT(rz_log, 8);
    return 16 << rz_log;
}

inline constexpr uptr ComputeRZLog(uptr user_requested_size) {
    u32 rz_log = user_requested_size <= 64 - 16            ? 0
                 : user_requested_size <= 128 - 32         ? 1
                 : user_requested_size <= 512 - 64         ? 2
                 : user_requested_size <= 4096 - 128       ? 3
                 : user_requested_size <= (1 << 14) - 256  ? 4
                 : user_requested_size <= (1 << 15) - 512  ? 5
                 : user_requested_size <= (1 << 16) - 1024 ? 6
                                                           : 7;
    return rz_log;
}

// ================================================================

static auto getUrResultString = [](ur_result_t Result) {
    switch (Result) {
    case UR_RESULT_SUCCESS:
        return "UR_RESULT_SUCCESS";
    case UR_RESULT_ERROR_INVALID_OPERATION:
        return "UR_RESULT_ERROR_INVALID_OPERATION";
    case UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES:
        return "UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES";
    case UR_RESULT_ERROR_INVALID_QUEUE:
        return "UR_RESULT_ERROR_INVALID_QUEUE";
    case UR_RESULT_ERROR_INVALID_VALUE:
        return "UR_RESULT_ERROR_INVALID_VALUE";
    case UR_RESULT_ERROR_INVALID_CONTEXT:
        return "UR_RESULT_ERROR_INVALID_CONTEXT";
    case UR_RESULT_ERROR_INVALID_PLATFORM:
        return "UR_RESULT_ERROR_INVALID_PLATFORM";
    case UR_RESULT_ERROR_INVALID_BINARY:
        return "UR_RESULT_ERROR_INVALID_BINARY";
    case UR_RESULT_ERROR_INVALID_PROGRAM:
        return "UR_RESULT_ERROR_INVALID_PROGRAM";
    case UR_RESULT_ERROR_INVALID_SAMPLER:
        return "UR_RESULT_ERROR_INVALID_SAMPLER";
    case UR_RESULT_ERROR_INVALID_BUFFER_SIZE:
        return "UR_RESULT_ERROR_INVALID_BUFFER_SIZE";
    case UR_RESULT_ERROR_INVALID_MEM_OBJECT:
        return "UR_RESULT_ERROR_INVALID_MEM_OBJECT";
    case UR_RESULT_ERROR_INVALID_EVENT:
        return "UR_RESULT_ERROR_INVALID_EVENT";
    case UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST:
        return "UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST";
    case UR_RESULT_ERROR_MISALIGNED_SUB_BUFFER_OFFSET:
        return "UR_RESULT_ERROR_MISALIGNED_SUB_BUFFER_OFFSET";
    case UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE:
        return "UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE";
    case UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE:
        return "UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE";
    case UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE:
        return "UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE";
    case UR_RESULT_ERROR_DEVICE_NOT_FOUND:
        return "UR_RESULT_ERROR_DEVICE_NOT_FOUND";
    case UR_RESULT_ERROR_INVALID_DEVICE:
        return "UR_RESULT_ERROR_INVALID_DEVICE";
    case UR_RESULT_ERROR_DEVICE_LOST:
        return "UR_RESULT_ERROR_DEVICE_LOST";
    case UR_RESULT_ERROR_DEVICE_REQUIRES_RESET:
        return "UR_RESULT_ERROR_DEVICE_REQUIRES_RESET";
    case UR_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE:
        return "UR_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE";
    case UR_RESULT_ERROR_DEVICE_PARTITION_FAILED:
        return "UR_RESULT_ERROR_DEVICE_PARTITION_FAILED";
    case UR_RESULT_ERROR_INVALID_DEVICE_PARTITION_COUNT:
        return "UR_RESULT_ERROR_INVALID_DEVICE_PARTITION_COUNT";
    case UR_RESULT_ERROR_INVALID_WORK_ITEM_SIZE:
        return "UR_RESULT_ERROR_INVALID_WORK_ITEM_SIZE";
    case UR_RESULT_ERROR_INVALID_WORK_DIMENSION:
        return "UR_RESULT_ERROR_INVALID_WORK_DIMENSION";
    case UR_RESULT_ERROR_INVALID_KERNEL_ARGS:
        return "UR_RESULT_ERROR_INVALID_KERNEL_ARGS";
    case UR_RESULT_ERROR_INVALID_KERNEL:
        return "UR_RESULT_ERROR_INVALID_KERNEL";
    case UR_RESULT_ERROR_INVALID_KERNEL_NAME:
        return "UR_RESULT_ERROR_INVALID_KERNEL_NAME";
    case UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
        return "UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
    case UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
        return "UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
    case UR_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
        return "UR_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
    case UR_RESULT_ERROR_INVALID_IMAGE_SIZE:
        return "UR_RESULT_ERROR_INVALID_IMAGE_SIZE";
    case UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return "UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case UR_RESULT_ERROR_IMAGE_FORMAT_NOT_SUPPORTED:
        return "UR_RESULT_ERROR_IMAGE_FORMAT_NOT_SUPPORTED";
    case UR_RESULT_ERROR_MEM_OBJECT_ALLOCATION_FAILURE:
        return "UR_RESULT_ERROR_MEM_OBJECT_ALLOCATION_FAILURE";
    case UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE:
        return "UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE";
    case UR_RESULT_ERROR_UNINITIALIZED:
        return "UR_RESULT_ERROR_UNINITIALIZED";
    case UR_RESULT_ERROR_OUT_OF_HOST_MEMORY:
        return "UR_RESULT_ERROR_OUT_OF_HOST_MEMORY";
    case UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
        return "UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
    case UR_RESULT_ERROR_OUT_OF_RESOURCES:
        return "UR_RESULT_ERROR_OUT_OF_RESOURCES";
    case UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE:
        return "UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE";
    case UR_RESULT_ERROR_PROGRAM_LINK_FAILURE:
        return "UR_RESULT_ERROR_PROGRAM_LINK_FAILURE";
    case UR_RESULT_ERROR_UNSUPPORTED_VERSION:
        return "UR_RESULT_ERROR_UNSUPPORTED_VERSION";
    case UR_RESULT_ERROR_UNSUPPORTED_FEATURE:
        return "UR_RESULT_ERROR_UNSUPPORTED_FEATURE";
    case UR_RESULT_ERROR_INVALID_ARGUMENT:
        return "UR_RESULT_ERROR_INVALID_ARGUMENT";
    case UR_RESULT_ERROR_INVALID_NULL_HANDLE:
        return "UR_RESULT_ERROR_INVALID_NULL_HANDLE";
    case UR_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
        return "UR_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
    case UR_RESULT_ERROR_INVALID_NULL_POINTER:
        return "UR_RESULT_ERROR_INVALID_NULL_POINTER";
    case UR_RESULT_ERROR_INVALID_SIZE:
        return "UR_RESULT_ERROR_INVALID_SIZE";
    case UR_RESULT_ERROR_UNSUPPORTED_SIZE:
        return "UR_RESULT_ERROR_UNSUPPORTED_SIZE";
    case UR_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
        return "UR_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
    case UR_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
        return "UR_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
    case UR_RESULT_ERROR_INVALID_ENUMERATION:
        return "UR_RESULT_ERROR_INVALID_ENUMERATION";
    case UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
        return "UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
    case UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
        return "UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
    case UR_RESULT_ERROR_INVALID_NATIVE_BINARY:
        return "UR_RESULT_ERROR_INVALID_NATIVE_BINARY";
    case UR_RESULT_ERROR_INVALID_GLOBAL_NAME:
        return "UR_RESULT_ERROR_INVALID_GLOBAL_NAME";
    case UR_RESULT_ERROR_INVALID_FUNCTION_NAME:
        return "UR_RESULT_ERROR_INVALID_FUNCTION_NAME";
    case UR_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
        return "UR_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
    case UR_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
        return "UR_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
    case UR_RESULT_ERROR_PROGRAM_UNLINKED:
        return "UR_RESULT_ERROR_PROGRAM_UNLINKED";
    case UR_RESULT_ERROR_OVERLAPPING_REGIONS:
        return "UR_RESULT_ERROR_OVERLAPPING_REGIONS";
    case UR_RESULT_ERROR_INVALID_HOST_PTR:
        return "UR_RESULT_ERROR_INVALID_HOST_PTR";
    case UR_RESULT_ERROR_INVALID_USM_SIZE:
        return "UR_RESULT_ERROR_INVALID_USM_SIZE";
    case UR_RESULT_ERROR_OBJECT_ALLOCATION_FAILURE:
        return "UR_RESULT_ERROR_OBJECT_ALLOCATION_FAILURE";
    case UR_RESULT_ERROR_ADAPTER_SPECIFIC:
        return "UR_RESULT_ERROR_ADAPTER_SPECIFIC";
    default:
        return "UR_RESULT_ERROR_UNKNOWN";
    }
};

// Trace an internal PI call; returns in case of an error.
#define UR_CALL(Call)                                                          \
    {                                                                          \
        if (PrintTrace)                                                        \
            context.logger.debug("UR ---> {}", #Call);                         \
        ur_result_t Result = (Call);                                           \
        if (PrintTrace)                                                        \
            context.logger.debug("UR <--- {}({})", #Call,                      \
                                 getUrResultString(Result));                   \
        if (Result != UR_RESULT_SUCCESS)                                       \
            return Result;                                                     \
    }

} // namespace ur_sanitizer_layer
