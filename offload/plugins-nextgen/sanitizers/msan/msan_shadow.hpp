//===--- msan_shadow.hpp - MSan shadow memory layout ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "msan_libdevice.hpp"
#include "sanitizer_common.hpp"

namespace llvm::offload::sanitizer {
namespace msan {

// clang-format off
/// Shadow Memory layout of GPU PVC device
///
/// USM Allocation Range (56 bits)
///   Host   USM : 0x00ff_f000_0000_0000 ~ 0x00ff_ffff_ffff_ffff
///   Shared USM : 0x0000_7000_0000_0000 ~ 0x0000_7fff_ffff_ffff
///   Device USM : 0xff00_0000_0000_0000 ~ 0xff00_ffff_ffff_ffff
///
/// Shadow Memory Mapping
///     0xff00_0000_0000_0000 - MSAN_SHADOW_BASE       : "invalid"
///     MSAN_SHADOW_BASE      - MSAN_SHADOW_END1       : "shadow-1" (MSAN_SHADOW_END1 - MSAN_SHADOW_BASE = 0x2000_0000_0000)
///     MSAN_SHADOW_END1      - MSAN_SHADOW_END2       : "origin-1" (MSAN_SHADOW_END1 - MSAN_SHADOW_END2 = 0x2000_0000_0000)
///     (gap)                                          :                                                  (0x1000_0000_0000)
///     MSAN_SHADOW_END3      - MSAN_SHADOW_END4       : "origin-2" (MSAN_SHADOW_END4 - MSAN_SHADOW_END3 = 0x5000_0000_0000)
///     MSAN_SHADOW_END4      - MSAN_SHADOW_END5       : "shadow-2" (MSAN_SHADOW_END5 - MSAN_SHADOW_END4 = 0x5000_0000_0000)
///     MSAN_SHADOW_END5      - 0xff00_ffff_ffff_ffff  : "app"      (MSAN_SHADOW_END5 - MSAN_SHADOW_BASE = 0xF000_0000_0000)
///
///  here,
///    - We assume "invalid" is not usable for user application (by observation)
///    - "shadow-1" and "origin-1" is use for host/shared USM
///    - "shadow-2" and "origin-2" is used for device USM
///    - "app" is device USM, the size of "app" is less than 0x5000_0000_0000_0000, so that it can be fully mapped to its shadow
///    - "gap" is necessary, so that "app" can be mapped to its shadow
// clang-format on
struct MsanShadowMemoryPVC {
  static bool IsDeviceUSM(uptr Ptr) { return Ptr >> 52 == 0xff0; }

  uptr MemToShadow(uptr Ptr) const {
    if (IsDeviceUSM(Ptr)) {
      return Ptr - 0x5000'0000'0000ULL;
    }
    // host/shared USM
    return (Ptr & 0xfff'ffff'ffffULL) + ((Ptr & 0x8000'0000'0000ULL) >> 3) +
           ShadowBegin;
  }

  uptr MemToOrigin(uptr Ptr) const {
    uptr AlignedPtr = RoundDownTo(Ptr, MSAN_ORIGIN_GRANULARITY);
    if (IsDeviceUSM(AlignedPtr)) {
      return AlignedPtr - 0xA000'0000'0000ULL;
    }
    // host/shared USM
    return (AlignedPtr & 0xfff'ffff'ffffULL) +
           ((AlignedPtr & 0x8000'0000'0000ULL) >> 3) + ShadowBegin +
           0x2000'0000'0000ULL;
  }

  static constexpr size_t GetShadowSize() { return 0xf000'0000'0000ULL; }
  static constexpr uptr GetStartAddress() { return 0x100'0000'0000'0000ULL; }

  uptr ShadowBegin = 0;
  uptr ShadowEnd = 0;
};

} // namespace msan
} // namespace llvm::offload::sanitizer
