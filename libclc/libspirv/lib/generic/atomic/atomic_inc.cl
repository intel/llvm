//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicIIncrement(local int *p, int scope,
                                                    int semantics) {
  return __sync_fetch_and_add(p, (int)1);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicIIncrement(global int *p, int scope,
                                                    int semantics) {
  return __sync_fetch_and_add(p, (int)1);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_AtomicIIncrement(local uint *p, int scope,
                                                     int semantics) {
  return __sync_fetch_and_add(p, (uint)1);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_AtomicIIncrement(global uint *p, int scope,
                                                     int semantics) {
  return __sync_fetch_and_add(p, (uint)1);
}

#ifdef cl_khr_int64_base_atomics
_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicIIncrement(local long *p, int scope,
                                                     int semantics) {
  return __sync_fetch_and_add(p, (long)1);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicIIncrement(global long *p, int scope,
                                                     int semantics) {
  return __sync_fetch_and_add(p, (long)1);
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_AtomicIIncrement(local ulong *p, int scope,
                                                      int semantics) {
  return __sync_fetch_and_add(p, (ulong)1);
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_AtomicIIncrement(global ulong *p,
                                                      int scope,
                                                      int semantics) {
  return __sync_fetch_and_add(p, (ulong)1);
}
#endif
