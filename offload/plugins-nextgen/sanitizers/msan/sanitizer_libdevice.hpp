//===--- sanitizer_libdevice.hpp - Host/device shared sanitizer types -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>

namespace llvm::offload::sanitizer {

enum class DeviceType : uint32_t { UNKNOWN = 0, CPU, GPU_PVC, GPU_DG2 };

enum class ErrorType : int32_t {
  UNKNOWN,
  OUT_OF_BOUNDS,
  MISALIGNED,
  USE_AFTER_FREE,
  OUT_OF_SHADOW_BOUNDS,
  UNKNOWN_DEVICE,
  NULL_POINTER,
};

} // namespace llvm::offload::sanitizer
