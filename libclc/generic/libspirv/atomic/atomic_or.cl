//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicOr(volatile global int *p,
                                            unsigned int scope,
                                            unsigned int semantics, int val) {
  return __sync_fetch_and_or(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int
__spirv_AtomicOr(volatile global unsigned int *p, unsigned int scope,
                 unsigned int semantics, unsigned int val) {
  return __sync_fetch_and_or(p, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicOr(volatile local int *p,
                                            unsigned int scope,
                                            unsigned int semantics, int val) {
  return __sync_fetch_and_or(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int
__spirv_AtomicOr(volatile local unsigned int *p, unsigned int scope,
                 unsigned int semantics, unsigned int val) {
  return __sync_fetch_and_or(p, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicOr(global int *p, unsigned int scope,
                                            unsigned int semantics, int val) {
  return __sync_fetch_and_or(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int __spirv_AtomicOr(global unsigned int *p,
                                                     unsigned int scope,
                                                     unsigned int semantics,
                                                     unsigned int val) {
  return __sync_fetch_and_or(p, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicOr(local int *p, unsigned int scope,
                                            unsigned int semantics, int val) {
  return __sync_fetch_and_or(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int __spirv_AtomicOr(local unsigned int *p,
                                                     unsigned int scope,
                                                     unsigned int semantics,
                                                     unsigned int val) {
  return __sync_fetch_and_or(p, val);
}

#ifdef cl_khr_int64_extended_atomics
_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicOr(volatile global long *p,
                                             unsigned int scope,
                                             unsigned int semantics, long val) {
  return __sync_fetch_and_or_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicOr(volatile global unsigned long *p, unsigned int scope,
                 unsigned int semantics, unsigned long val) {
  return __sync_fetch_and_or_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicOr(volatile local long *p,
                                             unsigned int scope,
                                             unsigned int semantics, long val) {
  return __sync_fetch_and_or_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicOr(volatile local unsigned long *p, unsigned int scope,
                 unsigned int semantics, unsigned long val) {
  return __sync_fetch_and_or_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicOr(global long *p, unsigned int scope,
                                             unsigned int semantics, long val) {
  return __sync_fetch_and_or_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long __spirv_AtomicOr(global unsigned long *p,
                                                      unsigned int scope,
                                                      unsigned int semantics,
                                                      unsigned long val) {
  return __sync_fetch_and_or_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicOr(local long *p, unsigned int scope,
                                             unsigned int semantics, long val) {
  return __sync_fetch_and_or_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long __spirv_AtomicOr(local unsigned long *p,
                                                      unsigned int scope,
                                                      unsigned int semantics,
                                                      unsigned long val) {
  return __sync_fetch_and_or_8(p, val);
}
#endif // cl_khr_int64_extended_atomics
