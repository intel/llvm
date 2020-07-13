//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicSMax(volatile global int *p,
                                              unsigned int scope,
                                              unsigned int semantics, int val) {
  return __sync_fetch_and_max(p, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicSMax(volatile local int *p,
                                              unsigned int scope,
                                              unsigned int semantics, int val) {
  return __sync_fetch_and_max(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int
__spirv_AtomicUMax(volatile global unsigned int *p, unsigned int scope,
                   unsigned int semantics, unsigned int val) {
  return __sync_fetch_and_umax(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int
__spirv_AtomicUMax(volatile local unsigned int *p, unsigned int scope,
                   unsigned int semantics, unsigned int val) {
  return __sync_fetch_and_umax(p, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicSMax(global int *p, unsigned int scope,
                                              unsigned int semantics, int val) {
  return __sync_fetch_and_max(p, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicSMax(local int *p, unsigned int scope,
                                              unsigned int semantics, int val) {
  return __sync_fetch_and_max(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int __spirv_AtomicUMax(global unsigned int *p,
                                                       unsigned int scope,
                                                       unsigned int semantics,
                                                       unsigned int val) {
  return __sync_fetch_and_umax(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int __spirv_AtomicUMax(local unsigned int *p,
                                                       unsigned int scope,
                                                       unsigned int semantics,
                                                       unsigned int val) {
  return __sync_fetch_and_umax(p, val);
}

#ifdef cl_khr_int64_extended_atomics
_CLC_DEF long __clc__sync_fetch_and_max_local_8(local long *, long);
_CLC_DEF long __clc__sync_fetch_and_max_global_8(global long *, long);
_CLC_DEF unsigned long __clc__sync_fetch_and_umax_local_8(local unsigned long *,
                                                          unsigned long);
_CLC_DEF unsigned long
__clc__sync_fetch_and_umax_global_8(global unsigned long *, unsigned long);

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicSMax(volatile global long *p,
                                               unsigned int scope,
                                               unsigned int semantics,
                                               long val) {
  return __clc__sync_fetch_and_max_global_8(p, val);
}
_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicSMax(volatile local long *p,
                                               unsigned int scope,
                                               unsigned int semantics,
                                               long val) {
  return __clc__sync_fetch_and_max_local_8(p, val);
}
_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicUMax(volatile global unsigned long *p, unsigned int scope,
                   unsigned int semantics, unsigned long val) {
  return __clc__sync_fetch_and_umax_global_8(p, val);
}
_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicUMax(volatile local unsigned long *p, unsigned int scope,
                   unsigned int semantics, unsigned long val) {
  return __clc__sync_fetch_and_umax_local_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicSMax(global long *p,
                                               unsigned int scope,
                                               unsigned int semantics,
                                               long val) {
  return __clc__sync_fetch_and_max_global_8(p, val);
}
_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicSMax(local long *p,
                                               unsigned int scope,
                                               unsigned int semantics,
                                               long val) {
  return __clc__sync_fetch_and_max_local_8(p, val);
}
_CLC_OVERLOAD _CLC_DEF unsigned long __spirv_AtomicUMax(global unsigned long *p,
                                                        unsigned int scope,
                                                        unsigned int semantics,
                                                        unsigned long val) {
  return __clc__sync_fetch_and_umax_global_8(p, val);
}
_CLC_OVERLOAD _CLC_DEF unsigned long __spirv_AtomicUMax(local unsigned long *p,
                                                        unsigned int scope,
                                                        unsigned int semantics,
                                                        unsigned long val) {
  return __clc__sync_fetch_and_umax_local_8(p, val);
}
#endif // cl_khr_int64_extended_atomics
