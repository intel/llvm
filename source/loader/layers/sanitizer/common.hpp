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

using uptr = uintptr_t;
using u8 = unsigned char;
using u32 = unsigned int;

constexpr unsigned ASAN_SHADOW_SCALE = 3;
constexpr unsigned ASAN_SHADOW_GRANULARITY = 1ULL << ASAN_SHADOW_SCALE;

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
inline constexpr uptr RZLog2Size(uptr rz_log) {
    assert(rz_log < 8);
    return 16 << rz_log;
}

inline constexpr uptr ComputeRZLog(uptr user_requested_size) {
    uptr rz_log = user_requested_size <= 64 - 16            ? 0
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

// Trace an internal UR call; returns in case of an error.
#define UR_CALL(Call)                                                          \
    {                                                                          \
        if (PrintTrace)                                                        \
            context.logger.debug("UR ---> {}", #Call);                         \
        ur_result_t Result = (Call);                                           \
        if (PrintTrace)                                                        \
            context.logger.debug("UR <--- {}({})", #Call, Result);             \
        if (Result != UR_RESULT_SUCCESS)                                       \
            return Result;                                                     \
    }

#ifndef NDEBUG
#define UR_ASSERT_EQ(Call, Result) assert(Call == Result)
#else
#define UR_ASSERT_EQ(Call, Result) (void)Call
#endif

} // namespace ur_sanitizer_layer
