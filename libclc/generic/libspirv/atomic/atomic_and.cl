//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicAnd(volatile global int *p,
                                             unsigned int scope,
                                             unsigned int semantics, int val) {
  return __sync_fetch_and_and(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int
__spirv_AtomicAnd(volatile global unsigned int *p, unsigned int scope,
                  unsigned int semantics, unsigned int val) {
  return __sync_fetch_and_and(p, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicAnd(volatile local int *p,
                                             unsigned int scope,
                                             unsigned int semantics, int val) {
  return __sync_fetch_and_and(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int
__spirv_AtomicAnd(volatile local unsigned int *p, unsigned int scope,
                  unsigned int semantics, unsigned int val) {
  return __sync_fetch_and_and(p, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicAnd(global int *p, unsigned int scope,
                                             unsigned int semantics, int val) {
  return __sync_fetch_and_and(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int __spirv_AtomicAnd(global unsigned int *p,
                                                      unsigned int scope,
                                                      unsigned int semantics,
                                                      unsigned int val) {
  return __sync_fetch_and_and(p, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicAnd(local int *p, unsigned int scope,
                                             unsigned int semantics, int val) {
  return __sync_fetch_and_and(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int __spirv_AtomicAnd(local unsigned int *p,
                                                      unsigned int scope,
                                                      unsigned int semantics,
                                                      unsigned int val) {
  return __sync_fetch_and_and(p, val);
}

#ifdef cl_khr_int64_extended_atomics
_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicAnd(volatile global long *p,
                                              unsigned int scope,
                                              unsigned int semantics,
                                              long val) {
  return __sync_fetch_and_and_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicAnd(volatile global unsigned long *p, unsigned int scope,
                  unsigned int semantics, unsigned long val) {
  return __sync_fetch_and_and_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicAnd(volatile local long *p,
                                              unsigned int scope,
                                              unsigned int semantics,
                                              long val) {
  return __sync_fetch_and_and_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicAnd(volatile local unsigned long *p, unsigned int scope,
                  unsigned int semantics, unsigned long val) {
  return __sync_fetch_and_and_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicAnd(global long *p,
                                              unsigned int scope,
                                              unsigned int semantics,
                                              long val) {
  return __sync_fetch_and_and_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long __spirv_AtomicAnd(global unsigned long *p,
                                                       unsigned int scope,
                                                       unsigned int semantics,
                                                       unsigned long val) {
  return __sync_fetch_and_and_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicAnd(local long *p, unsigned int scope,
                                              unsigned int semantics,
                                              long val) {
  return __sync_fetch_and_and_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long __spirv_AtomicAnd(local unsigned long *p,
                                                       unsigned int scope,
                                                       unsigned int semantics,
                                                       unsigned long val) {
  return __sync_fetch_and_and_8(p, val);
}
#endif // cl_khr_int64_base_atomics
