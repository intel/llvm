//==-- device-sanitizer-report.hpp - Structure and declaration for assert
// support --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

// Treat this header as system one to workaround frontend's restriction
#pragma clang system_header

#include <cinttypes>

enum class DeviceSanitizerErrorType : int32_t {
  UNKNOWN,
  OUT_OF_BOUNDS,
  MISALIGNED,
  USE_AFTER_FREE,
  OUT_OF_SHADOW_BOUNDS,
};

enum class DeviceSanitizerMemoryType : int32_t {
  UNKNOWN,
  USM_DEVICE,
  USM_HOST,
  USM_SHARED,
  LOCAL,
  PRIVATE,
  MEM_BUFFER,
};

// NOTE Layout of this structure should be aligned with the one in
// sycl/include/sycl/detail/device_sanitizer_report.hpp
struct DeviceSanitizerReport {
  int Flag = 0;

  char File[256 + 1] = "";
  char Func[256 + 1] = "";

  int32_t Line = 0;

  uint64_t GID0 = 0;
  uint64_t GID1 = 0;
  uint64_t GID2 = 0;

  uint64_t LID0 = 0;
  uint64_t LID1 = 0;
  uint64_t LID2 = 0;

  uint64_t Address = 0;
  bool IsWrite = false;
  uint32_t AccessSize = 0;
  DeviceSanitizerMemoryType MemoryType = DeviceSanitizerMemoryType::UNKNOWN;
  DeviceSanitizerErrorType ErrorType = DeviceSanitizerErrorType::UNKNOWN;

  bool IsRecover = false;
};
