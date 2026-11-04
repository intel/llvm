//===--- sanitizer_common.hpp - Device sanitizer common helpers -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <cstdint>
#include <string>

namespace llvm::offload::sanitizer {

// ================================================================
// Copy from LLVM compiler-rt/lib/asan

using uptr = uintptr_t;
using u8 = unsigned char;

inline constexpr bool IsPowerOfTwo(uptr x) {
  return (x & (x - 1)) == 0 && x != 0;
}

inline constexpr uptr RoundDownTo(uptr x, uptr boundary) {
  assert(IsPowerOfTwo(boundary));
  return x & ~(boundary - 1);
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

std::string DemangleName(const std::string &name);

// Sync with llvm/Transforms/Instrumentation/SPIRVSanitizerCommonUtils.h
enum SanitizedKernelFlags : uint32_t {
  CHECK_LOCALS = 1U << 2,
  CHECK_PRIVATES = 1U << 3,
  MSAN_TRACK_ORIGINS = 1U << 5,
};

} // namespace llvm::offload::sanitizer
