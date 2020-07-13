//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicISub(volatile global int *p,
                                              unsigned int scope,
                                              unsigned int semantics, int val) {
  return __sync_fetch_and_sub(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int
__spirv_AtomicISub(volatile global unsigned int *p, unsigned int scope,
                   unsigned int semantics, unsigned int val) {
  return __sync_fetch_and_sub(p, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicISub(volatile local int *p,
                                              unsigned int scope,
                                              unsigned int semantics, int val) {
  return __sync_fetch_and_sub(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int
__spirv_AtomicISub(volatile local unsigned int *p, unsigned int scope,
                   unsigned int semantics, unsigned int val) {
  return __sync_fetch_and_sub(p, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicISub(global int *p, unsigned int scope,
                                              unsigned int semantics, int val) {
  return __sync_fetch_and_sub(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int __spirv_AtomicISub(global unsigned int *p,
                                                       unsigned int scope,
                                                       unsigned int semantics,
                                                       unsigned int val) {
  return __sync_fetch_and_sub(p, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicISub(local int *p, unsigned int scope,
                                              unsigned int semantics, int val) {
  return __sync_fetch_and_sub(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int __spirv_AtomicISub(local unsigned int *p,
                                                       unsigned int scope,
                                                       unsigned int semantics,
                                                       unsigned int val) {
  return __sync_fetch_and_sub(p, val);
}

#ifdef cl_khr_int64_extended_atomics
_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicISub(volatile global long *p,
                                               unsigned int scope,
                                               unsigned int semantics,
                                               long val) {
  return __sync_fetch_and_sub_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicISub(volatile global unsigned long *p, unsigned int scope,
                   unsigned int semantics, unsigned long val) {
  return __sync_fetch_and_sub_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicISub(volatile local long *p,
                                               unsigned int scope,
                                               unsigned int semantics,
                                               long val) {
  return __sync_fetch_and_sub_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicISub(volatile local unsigned long *p, unsigned int scope,
                   unsigned int semantics, unsigned long val) {
  return __sync_fetch_and_sub_8(p, val);
}
_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicISub(global long *p,
                                               unsigned int scope,
                                               unsigned int semantics,
                                               long val) {
  return __sync_fetch_and_sub_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long __spirv_AtomicISub(global unsigned long *p,
                                                        unsigned int scope,
                                                        unsigned int semantics,
                                                        unsigned long val) {
  return __sync_fetch_and_sub_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicISub(local long *p,
                                               unsigned int scope,
                                               unsigned int semantics,
                                               long val) {
  return __sync_fetch_and_sub_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long __spirv_AtomicISub(local unsigned long *p,
                                                        unsigned int scope,
                                                        unsigned int semantics,
                                                        unsigned long val) {
  return __sync_fetch_and_sub_8(p, val);
}
#endif // cl_khr_int64_extended_atomics
