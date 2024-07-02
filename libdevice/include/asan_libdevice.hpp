//===---- asan_libdevice.hpp - Structure and declaration for sanitizer ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cinttypes>

// NOTE This file should be sync with
// unified-runtime/source/loader/layers/sanitizer/device_sanitizer_report.hpp

enum class DeviceSanitizerErrorType : int32_t {
  UNKNOWN,
  OUT_OF_BOUNDS,
  MISALIGNED,
  USE_AFTER_FREE,
  OUT_OF_SHADOW_BOUNDS,
  UNKNOWN_DEVICE,
  NULL_POINTER,
};

enum class DeviceSanitizerMemoryType : int32_t {
  UNKNOWN,
  USM_DEVICE,
  USM_HOST,
  USM_SHARED,
  LOCAL,
  PRIVATE,
  MEM_BUFFER,
  DEVICE_GLOBAL,
};

struct DeviceSanitizerReport {
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
  DeviceSanitizerMemoryType MemoryType = DeviceSanitizerMemoryType::UNKNOWN;
  DeviceSanitizerErrorType ErrorType = DeviceSanitizerErrorType::UNKNOWN;

  bool IsRecover = false;
};

struct LocalArgsInfo {
  uint64_t Size = 0;
  uint64_t SizeWithRedZone = 0;
};

constexpr std::size_t ASAN_MAX_NUM_REPORTS = 10;

struct LaunchInfo {
  uintptr_t PrivateShadowOffset = 0;
  uintptr_t PrivateShadowOffsetEnd = 0;

  uintptr_t LocalShadowOffset = 0;
  uintptr_t LocalShadowOffsetEnd = 0;

  uint32_t NumLocalArgs = 0;
  LocalArgsInfo *LocalArgs = nullptr; // Ordered by ArgIndex

  DeviceSanitizerReport SanitizerReport[ASAN_MAX_NUM_REPORTS];
};

constexpr unsigned ASAN_SHADOW_SCALE = 4;
constexpr unsigned ASAN_SHADOW_GRANULARITY = 1ULL << ASAN_SHADOW_SCALE;

// Based on the observation, only the last 24 bits of the address of the private
// variable have changed
constexpr std::size_t ASAN_PRIVATE_SIZE = 0xffffffULL + 1;

// These magic values are written to shadow for better error
// reporting.
constexpr int kUsmDeviceRedzoneMagic = (char)0x81;
constexpr int kUsmHostRedzoneMagic = (char)0x82;
constexpr int kUsmSharedRedzoneMagic = (char)0x83;
constexpr int kMemBufferRedzoneMagic = (char)0x84;
constexpr int kDeviceGlobalRedzoneMagic = (char)0x85;
constexpr int kNullPointerRedzoneMagic = (char)0x86;

constexpr int kUsmDeviceDeallocatedMagic = (char)0x91;
constexpr int kUsmHostDeallocatedMagic = (char)0x92;
constexpr int kUsmSharedDeallocatedMagic = (char)0x93;
constexpr int kMemBufferDeallocatedMagic = (char)0x93;

constexpr int kSharedLocalRedzoneMagic = (char)0xa1;

// Same with host ASan stack
const int kPrivateLeftRedzoneMagic = (char)0xf1;
const int kPrivateMidRedzoneMagic = (char)0xf2;
const int kPrivateRightRedzoneMagic = (char)0xf3;

constexpr auto kSPIR_AsanShadowMemoryGlobalStart =
    "__AsanShadowMemoryGlobalStart";
constexpr auto kSPIR_AsanShadowMemoryGlobalEnd = "__AsanShadowMemoryGlobalEnd";

constexpr auto kSPIR_DeviceType = "__DeviceType";
constexpr auto kSPIR_AsanDebug = "__AsanDebug";

constexpr auto kSPIR_AsanDeviceGlobalCount = "__AsanDeviceGlobalCount";
constexpr auto kSPIR_AsanDeviceGlobalMetadata = "__AsanDeviceGlobalMetadata";

inline const char *ToString(DeviceSanitizerMemoryType MemoryType) {
  switch (MemoryType) {
  case DeviceSanitizerMemoryType::USM_DEVICE:
    return "Device USM";
  case DeviceSanitizerMemoryType::USM_HOST:
    return "Host USM";
  case DeviceSanitizerMemoryType::USM_SHARED:
    return "Shared USM";
  case DeviceSanitizerMemoryType::LOCAL:
    return "Local Memory";
  case DeviceSanitizerMemoryType::PRIVATE:
    return "Private Memory";
  case DeviceSanitizerMemoryType::MEM_BUFFER:
    return "Memory Buffer";
  case DeviceSanitizerMemoryType::DEVICE_GLOBAL:
    return "Device Global";
  default:
    return "Unknown Memory";
  }
}

inline const char *ToString(DeviceSanitizerErrorType ErrorType) {
  switch (ErrorType) {
  case DeviceSanitizerErrorType::OUT_OF_BOUNDS:
    return "out-of-bounds-access";
  case DeviceSanitizerErrorType::MISALIGNED:
    return "misaligned-access";
  case DeviceSanitizerErrorType::USE_AFTER_FREE:
    return "use-after-free";
  case DeviceSanitizerErrorType::OUT_OF_SHADOW_BOUNDS:
    return "out-of-shadow-bounds-access";
  case DeviceSanitizerErrorType::UNKNOWN_DEVICE:
    return "unknown-device";
  case DeviceSanitizerErrorType::NULL_POINTER:
    return "null-pointer-access";
  default:
    return "unknown-error";
  }
}
