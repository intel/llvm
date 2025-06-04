//==-- tsan_rtl.hpp - Declaration for sanitizer global var ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "sanitizer_defs.hpp"
#include "spir_global_var.hpp"
#include "tsan/tsan_libdevice.hpp"

// Treat this header as system one to workaround frontend's restriction
#pragma clang system_header

#if defined(__SPIR__) || defined(__SPIRV__)

struct Shadow {
public:
  static constexpr RawShadow kEmpty = static_cast<RawShadow>(0);

  // A marker to indicate that the current address will not trigger race
  // condition.
  static constexpr RawShadow kRodata = static_cast<RawShadow>(1 << 30);

  Shadow(uint32_t addr, uint32_t size, uint32_t clock, uint32_t sid,
         AccessType typ) {
    raw_ = 0;
    raw_ |= (!!(typ & kAccessAtomic) << 31) | (!!(typ & kAccessRead) << 30) |
            (clock << 16) | (sid << 8) |
            (((((1u << size) - 1) << (addr & 0x7)) & 0xff));
  }

  explicit Shadow(RawShadow x = Shadow::kEmpty) {
    raw_ = static_cast<uint32_t>(x);
  }

  RawShadow raw() const { return static_cast<RawShadow>(raw_); }

  Sid sid() const { return part_.sid_; }

  uint16_t clock() const { return part_.clock_; }

  uint8_t access() const { return part_.access_; }

  bool IsBothReads(AccessType Type) {
    uint32_t is_read = !!(Type & kAccessRead);
    bool res = raw_ & (is_read << 30);
    return res;
  }

private:
  struct Parts {
    uint8_t access_;
    Sid sid_;
    uint16_t clock_ : 14;
    uint16_t is_read_ : 1;
    uint16_t is_atomic_ : 1;
  };
  union {
    Parts part_;
    uint32_t raw_;
  };
};

#endif // __SPIR__ || __SPIRV__
