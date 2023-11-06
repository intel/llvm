//==--- sanitizer_utils.cpp - device sanitizer util inserted by compiler ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device.h"
#include <cstdint>

using uptr = uint64_t;
#if defined(__SPIR__)
// TODO: add real implementation in __asan_load_n.
DEVICE_EXTERN_C_NOINLINE
void __asan_load_n(uptr addr, unsigned n) {
  (void)addr;
  (void)n;
  return;
}

DEVICE_EXTERN_C_NOINLINE
void __asan_load4(uptr addr) { __asan_load_n(addr, 4); }

DEVICE_EXTERN_C_NOINLINE
void __asan_load8(uptr addr) { __asan_load_n(addr, 8); }
#endif // __SPIR__
