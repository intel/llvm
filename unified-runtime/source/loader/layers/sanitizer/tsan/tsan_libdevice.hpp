/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file tsan_libdevice.hpp
 *
 */

#pragma once

#include "sanitizer_common/sanitizer_libdevice.hpp"

#if !defined(__SPIR__) && !defined(__SPIRV__)
namespace ur_sanitizer_layer {
#endif // !__SPIR__ && !__SPIRV__

// Thread slot ID.
typedef uint8_t Sid;

// Abstract time unit, vector clock element.
typedef int Epoch;

const uintptr_t kThreadSlotCount = 256;

// Count of shadow values in a shadow cell.
const uintptr_t kShadowCnt = 2;

// That many user bytes are mapped onto a single shadow cell.
const uintptr_t kShadowCell = 8;

// Single shadow value.
typedef uint32_t RawShadow;
const uint64_t kShadowSize = sizeof(RawShadow);

// Shadow memory is kShadowMultiplier times larger than user memory.
const uintptr_t kShadowMultiplier = kShadowSize * kShadowCnt / kShadowCell;

typedef uint32_t AccessType;

enum : AccessType {
  kAccessWrite = 0,
  kAccessRead = 1 << 0,
  kAccessAtomic = 1 << 1,
};

// Fixed-size vector clock, used both for threads and sync objects.
struct VectorClock {
  Epoch clk_[kThreadSlotCount] = {};
};

struct TsanErrorReport {
  int Flag = 0;

  char File[512 + 1] = {};
  char Func[512 + 1] = {};

  int32_t Line = 0;

  uint64_t GID0 = 0;
  uint64_t GID1 = 0;
  uint64_t GID2 = 0;

  uint64_t LID0 = 0;
  uint64_t LID1 = 0;
  uint64_t LID2 = 0;

  uintptr_t Address = 0;
  AccessType Type;
  uint32_t AccessSize = 0;
};

constexpr uint64_t TSAN_MAX_NUM_REPORTS = 128;

struct TsanRuntimeData {
  uintptr_t GlobalShadowOffset = 0;

  uintptr_t GlobalShadowOffsetEnd = 0;

  // The last one is to record global state
  VectorClock Clock[kThreadSlotCount + 1];

  DeviceType DeviceTy = DeviceType::UNKNOWN;

  uint32_t Debug = 0;

  int Lock = 0;

  uint32_t RecordedReportCount = 0;

  TsanErrorReport Report[TSAN_MAX_NUM_REPORTS];
};

constexpr auto kSPIR_TsanDeviceGlobalMetadata = "__TsanDeviceGlobalMetadata";

constexpr auto kSPIR_TsanSpirKernelMetadata = "__TsanKernelMetadata";

#if !defined(__SPIR__) && !defined(__SPIRV__)
} // namespace ur_sanitizer_layer
#endif // !__SPIR__ && !__SPIRV__
