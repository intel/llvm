/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer_common.hpp
 *
 */

#pragma once

#include "ur/ur.hpp"
#include "ur_ddi.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>

namespace ur_sanitizer_layer {

// ================================================================
// Copy from LLVM compiler-rt/lib/asan

using uptr = uintptr_t;
using u8 = unsigned char;
using u32 = unsigned int;

inline constexpr bool IsPowerOfTwo(uptr x) {
  return (x & (x - 1)) == 0 && x != 0;
}

inline constexpr uptr RoundUpToPowerOfTwo(uptr x) {
  if (x == 0)
    return 1;
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  x++;
  assert(IsPowerOfTwo(x));
  return x;
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

// Valid redzone sizes are 16, 32, 64 ..., so we encode them in 3 bits.
// We use adaptive redzones: for larger allocation larger redzones are used.
inline constexpr uptr RZLog2Size(uptr rz_log) { return 16 << rz_log; }

inline constexpr uptr RZSize2Log(uptr rz_size) {
  assert(rz_size >= 16);
  assert(IsPowerOfTwo(rz_size));
  uptr res = log2(rz_size) - 4;
  assert(rz_size == RZLog2Size(res));
  return res;
}

inline constexpr uptr ComputeRZLog(uptr user_requested_size, uptr min_size) {
  uptr rz_log = user_requested_size <= 64 - 16            ? 0
                : user_requested_size <= 128 - 32         ? 1
                : user_requested_size <= 512 - 64         ? 2
                : user_requested_size <= 4096 - 128       ? 3
                : user_requested_size <= (1 << 14) - 256  ? 4
                : user_requested_size <= (1 << 15) - 512  ? 5
                : user_requested_size <= (1 << 16) - 1024 ? 6
                                                          : 7;
  uptr min_log = RZSize2Log(min_size);
  return std::max(rz_log, min_log);
}

/// Returns the next integer (mod 2**64) that is greater than or equal to
/// \p Value and is a multiple of \p Align. \p Align must be non-zero.
///
/// Examples:
/// \code
///   alignTo(5, 8) = 8
///   alignTo(17, 8) = 24
///   alignTo(~0LL, 8) = 0
///   alignTo(321, 255) = 510
/// \endcode
inline uint64_t AlignTo(uint64_t Value, uint64_t Align) {
  assert(Align != 0u && "Align can't be 0.");
  return (Value + Align - 1) / Align * Align;
}

inline uint64_t GetSizeAndRedzoneSizeForLocal(uint64_t Size,
                                              uint64_t Granularity,
                                              uint64_t Alignment) {
  uint64_t Res = 0;
  if (Size <= 4) {
    Res = 16;
  } else if (Size <= 16) {
    Res = 32;
  } else if (Size <= 128) {
    Res = Size + 32;
  } else if (Size <= 512) {
    Res = Size + 64;
  } else if (Size <= 4096) {
    Res = Size + 128;
  } else {
    Res = Size + 256;
  }
  return AlignTo(std::max(Res, 2 * Granularity), Alignment);
}

// ================================================================

using BacktraceFrame = void *;
using BacktraceInfo = std::string;

struct SourceInfo {
  std::string file;
  std::string function;
  int line = 0;
  int column = 0;
};

bool IsInASanContext();

uptr MmapFixedNoReserve(uptr Addr, uptr Size);
uptr MmapNoReserve(uptr Addr, uptr Size);
bool Munmap(uptr Addr, uptr Size);
uptr ProtectMemoryRange(uptr Addr, uptr Size);
bool DontCoredumpRange(uptr Addr, uptr Size);

void *GetMemFunctionPointer(const char *);

std::string DemangleName(const std::string &name);

// ================================================================

// Sync with llvm/Transforms/Instrumentation/SPIRVSanitizerCommonUtils.h
enum SanitizedKernelFlags : uint32_t {
  NO_CHECK = 0,
  CHECK_GLOBALS = 1U << 1,
  CHECK_LOCALS = 1U << 2,
  CHECK_PRIVATES = 1U << 3,
  CHECK_GENERICS = 1U << 4,
  MSAN_TRACK_ORIGINS = 1U << 5,
};

} // namespace ur_sanitizer_layer
