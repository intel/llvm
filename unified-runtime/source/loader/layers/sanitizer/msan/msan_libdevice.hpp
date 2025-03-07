/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file msan_libdevice.hpp
 *
 */

#pragma once

#include "sanitizer_common/sanitizer_libdevice.hpp"

#if !defined(__SPIR__) && !defined(__SPIRV__)
namespace ur_sanitizer_layer {
#endif // !__SPIR__ && !__SPIRV__

struct MsanErrorReport {
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

  uint32_t AccessSize = 0;
  ErrorType ErrorTy = ErrorType::UNKNOWN;
};

struct MsanLocalArgsInfo {
  uint64_t Size = 0;
  uint64_t SizeWithRedZone = 0;
};

struct MsanLaunchInfo {
  uintptr_t GlobalShadowOffset = 0;
  uintptr_t GlobalShadowOffsetEnd = 0;

  uintptr_t LocalShadowOffset = 0;
  uintptr_t LocalShadowOffsetEnd = 0;

  uintptr_t CleanShadow = 0;

  DeviceType DeviceTy = DeviceType::UNKNOWN;
  uint32_t Debug = 0;
  uint32_t IsRecover = 0;

  MsanErrorReport Report;
};

// Based on the observation, only the last 24 bits of the address of the private
// variable have changed
constexpr std::size_t MSAN_PRIVATE_SIZE = 0xffffffULL + 1;

constexpr auto kSPIR_MsanDeviceGlobalMetadata = "__MsanDeviceGlobalMetadata";
constexpr auto kSPIR_MsanSpirKernelMetadata = "__MsanKernelMetadata";

#if !defined(__SPIR__) && !defined(__SPIRV__)
} // namespace ur_sanitizer_layer
#endif // !__SPIR__ && !__SPIRV__
