/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_libdevice.hpp
 *
 */

#pragma once

#include "sanitizer_common/sanitizer_libdevice.hpp"

#if !defined(__SPIR__) && !defined(__SPIRV__)
namespace ur_sanitizer_layer {
#endif // !__SPIR__ && !__SPIRV__

struct AsanErrorReport {
  int Flag = 0;

  char File[256 + 1] = {};
  char Func[256 + 1] = {};

  int32_t Line = 0;

  uint64_t GID0 = 0;
  uint64_t GID1 = 0;
  uint64_t GID2 = 0;

  uint64_t LID0 = 0;
  uint64_t LID1 = 0;
  uint64_t LID2 = 0;

  uintptr_t Address = 0;
  bool IsWrite = false;
  uint32_t AccessSize = 0;
  MemoryType MemoryTy = MemoryType::UNKNOWN;
  ErrorType ErrorTy = ErrorType::UNKNOWN;

  bool IsRecover = false;
};

struct LocalArgsInfo {
  uint64_t Size = 0;
  uint64_t SizeWithRedZone = 0;
};

constexpr uint32_t ASAN_MAX_WG_LOCAL = 8192;

constexpr uint32_t ASAN_MAX_SG_PRIVATE = 256;

constexpr uint64_t ASAN_MAX_NUM_REPORTS = 10;

struct AsanRuntimeData {
  uintptr_t GlobalShadowOffset = 0;
  uintptr_t GlobalShadowOffsetEnd = 0;

  uintptr_t GlobalShadowLowerBound = 0;
  uintptr_t GlobalShadowUpperBound = 0;

  uintptr_t *PrivateBase = nullptr;
  uintptr_t PrivateShadowOffset = 0;
  uintptr_t PrivateShadowOffsetEnd = 0;

  uintptr_t LocalShadowOffset = 0;
  uintptr_t LocalShadowOffsetEnd = 0;

  LocalArgsInfo *LocalArgs = nullptr; // Ordered by ArgIndex
  uint32_t NumLocalArgs = 0;

  DeviceType DeviceTy = DeviceType::UNKNOWN;
  uint32_t Debug = 0;

  int ReportFlag = 0;
  AsanErrorReport Report[ASAN_MAX_NUM_REPORTS] = {};
};

constexpr unsigned ASAN_SHADOW_SCALE = 4;
constexpr unsigned ASAN_SHADOW_GRANULARITY = 1ULL << ASAN_SHADOW_SCALE;

// Based on the observation, only the last 24 bits of the address of the private
// variable have changed
constexpr uint64_t ASAN_PRIVATE_SIZE = 0xffffffULL + 1;

// These magic values are written to shadow for better error
// reporting.
constexpr int8_t kUsmDeviceRedzoneMagic = (int8_t)0x81;
constexpr int8_t kUsmHostRedzoneMagic = (int8_t)0x82;
constexpr int8_t kUsmSharedRedzoneMagic = (int8_t)0x83;
constexpr int8_t kMemBufferRedzoneMagic = (int8_t)0x84;
constexpr int8_t kDeviceGlobalRedzoneMagic = (int8_t)0x85;
constexpr int8_t kNullPointerRedzoneMagic = (int8_t)0x86;

constexpr int8_t kUsmDeviceDeallocatedMagic = (int8_t)0x91;
constexpr int8_t kUsmHostDeallocatedMagic = (int8_t)0x92;
constexpr int8_t kUsmSharedDeallocatedMagic = (int8_t)0x93;
constexpr int8_t kMemBufferDeallocatedMagic = (int8_t)0x93;

constexpr int8_t kSharedLocalRedzoneMagic = (int8_t)0xa1;

// Same with host ASan stack
const int8_t kPrivateLeftRedzoneMagic = (int8_t)0xf1;
const int8_t kPrivateMidRedzoneMagic = (int8_t)0xf2;
const int8_t kPrivateRightRedzoneMagic = (int8_t)0xf3;

// Unknown shadow value
constexpr int8_t kUnknownMagic = (int8_t)0xff;

constexpr auto kSPIR_AsanDeviceGlobalMetadata = "__AsanDeviceGlobalMetadata";
constexpr auto kSPIR_AsanSpirKernelMetadata = "__AsanKernelMetadata";

#if !defined(__SPIR__) && !defined(__SPIRV__)
} // namespace ur_sanitizer_layer
#endif // !__SPIR__ && !__SPIRV__
