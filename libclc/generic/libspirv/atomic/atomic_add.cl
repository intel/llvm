//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicIAdd(volatile global int *p,
                                              unsigned int scope,
                                              unsigned int semantics, int val) {
  return __sync_fetch_and_add(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int
__spirv_AtomicIAdd(volatile global unsigned int *p, unsigned int scope,
                   unsigned int semantics, unsigned int val) {
  return __sync_fetch_and_add(p, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicIAdd(volatile local int *p,
                                              unsigned int scope,
                                              unsigned int semantics, int val) {
  return __sync_fetch_and_add(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int
__spirv_AtomicIAdd(volatile local unsigned int *p, unsigned int scope,
                   unsigned int semantics, unsigned int val) {
  return __sync_fetch_and_add(p, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicIAdd(global int *p, unsigned int scope,
                                              unsigned int semantics, int val) {
  return __sync_fetch_and_add(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int __spirv_AtomicIAdd(global unsigned int *p,
                                                       unsigned int scope,
                                                       unsigned int semantics,
                                                       unsigned int val) {
  return __sync_fetch_and_add(p, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicIAdd(local int *p, unsigned int scope,
                                              unsigned int semantics, int val) {
  return __sync_fetch_and_add(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int __spirv_AtomicIAdd(local unsigned int *p,
                                                       unsigned int scope,
                                                       unsigned int semantics,
                                                       unsigned int val) {
  return __sync_fetch_and_add(p, val);
}

#ifdef cl_khr_int64_base_atomics
_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicIAdd(volatile global long int *p,
                                               unsigned int scope,
                                               unsigned int semantics,
                                               long val) {
  return __sync_fetch_and_add_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicIAdd(volatile global unsigned long int *p, unsigned int scope,
                   unsigned int semantics, unsigned long val) {
  return __sync_fetch_and_add_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicIAdd(volatile local long int *p,
                                               unsigned int scope,
                                               unsigned int semantics,
                                               long val) {
  return __sync_fetch_and_add_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicIAdd(volatile local unsigned long int *p, unsigned int scope,
                   unsigned int semantics, unsigned long val) {
  return __sync_fetch_and_add_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicIAdd(global long int *p,
                                               unsigned int scope,
                                               unsigned int semantics,
                                               long val) {
  return __sync_fetch_and_add_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicIAdd(global unsigned long int *p, unsigned int scope,
                   unsigned int semantics, unsigned long val) {
  return __sync_fetch_and_add_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicIAdd(local long int *p,
                                               unsigned int scope,
                                               unsigned int semantics,
                                               long val) {
  return __sync_fetch_and_add_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicIAdd(local unsigned long int *p, unsigned int scope,
                   unsigned int semantics, unsigned long val) {
  return __sync_fetch_and_add_8(p, val);
}
#endif // cl_khr_int64_base_atomics
