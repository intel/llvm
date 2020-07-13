//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicSMin(volatile global int *p,
                                              unsigned int scope,
                                              unsigned int semantics, int val) {
  return __sync_fetch_and_min(p, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicSMin(volatile local int *p,
                                              unsigned int scope,
                                              unsigned int semantics, int val) {
  return __sync_fetch_and_min(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int
__spirv_AtomicUMin(volatile global unsigned int *p, unsigned int scope,
                   unsigned int semantics, unsigned int val) {
  return __sync_fetch_and_umin(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int
__spirv_AtomicUMin(volatile local unsigned int *p, unsigned int scope,
                   unsigned int semantics, unsigned int val) {
  return __sync_fetch_and_umin(p, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicSMin(global int *p, unsigned int scope,
                                              unsigned int semantics, int val) {
  return __sync_fetch_and_min(p, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicSMin(local int *p, unsigned int scope,
                                              unsigned int semantics, int val) {
  return __sync_fetch_and_min(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int __spirv_AtomicUMin(global unsigned int *p,
                                                       unsigned int scope,
                                                       unsigned int semantics,
                                                       unsigned int val) {
  return __sync_fetch_and_umin(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int __spirv_AtomicUMin(local unsigned int *p,
                                                       unsigned int scope,
                                                       unsigned int semantics,
                                                       unsigned int val) {
  return __sync_fetch_and_umin(p, val);
}

#ifdef cl_khr_int64_extended_atomics
_CLC_DEF long __clc__sync_fetch_and_min_local_8(local long *, long);
_CLC_DEF long __clc__sync_fetch_and_min_global_8(global long *, long);
_CLC_DEF unsigned long __clc__sync_fetch_and_umin_local_8(local unsigned long *,
                                                          unsigned long);
_CLC_DEF unsigned long
__clc__sync_fetch_and_umin_global_8(global unsigned long *, unsigned long);

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicSMin(volatile global long *p,
                                               unsigned int scope,
                                               unsigned int semantics,
                                               long val) {
  return __clc__sync_fetch_and_min_global_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicSMin(volatile local long *p,
                                               unsigned int scope,
                                               unsigned int semantics,
                                               long val) {
  return __clc__sync_fetch_and_min_local_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicUMin(volatile global unsigned long *p, unsigned int scope,
                   unsigned int semantics, unsigned long val) {
  return __clc__sync_fetch_and_umin_global_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicUMin(volatile local unsigned long *p, unsigned int scope,
                   unsigned int semantics, unsigned long val) {
  return __clc__sync_fetch_and_umin_local_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicSMin(global long *p,
                                               unsigned int scope,
                                               unsigned int semantics,
                                               long val) {
  return __clc__sync_fetch_and_min_global_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicSMin(local long *p,
                                               unsigned int scope,
                                               unsigned int semantics,
                                               long val) {
  return __clc__sync_fetch_and_min_local_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long __spirv_AtomicUMin(global unsigned long *p,
                                                        unsigned int scope,
                                                        unsigned int semantics,
                                                        unsigned long val) {
  return __clc__sync_fetch_and_umin_global_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long __spirv_AtomicUMin(local unsigned long *p,
                                                        unsigned int scope,
                                                        unsigned int semantics,
                                                        unsigned long val) {
  return __clc__sync_fetch_and_umin_local_8(p, val);
}
#endif // cl_khr_int64_extended_atomics
