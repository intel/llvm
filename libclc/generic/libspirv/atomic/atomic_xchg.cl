//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicExchange(volatile global int *p,
                                                  unsigned int scope,
                                                  unsigned int semantics,
                                                  int val) {
  return __sync_swap_4(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int
__spirv_AtomicExchange(volatile global unsigned int *p, unsigned int scope,
                       unsigned int semantics, unsigned int val) {
  return __sync_swap_4(p, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicExchange(volatile local int *p,
                                                  unsigned int scope,
                                                  unsigned int semantics,
                                                  int val) {
  return __sync_swap_4(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int
__spirv_AtomicExchange(volatile local unsigned int *p, unsigned int scope,
                       unsigned int semantics, unsigned int val) {
  return __sync_swap_4(p, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicExchange(global int *p,
                                                  unsigned int scope,
                                                  unsigned int semantics,
                                                  int val) {
  return __sync_swap_4(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int
__spirv_AtomicExchange(global unsigned int *p, unsigned int scope,
                       unsigned int semantics, unsigned int val) {
  return __sync_swap_4(p, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicExchange(local int *p,
                                                  unsigned int scope,
                                                  unsigned int semantics,
                                                  int val) {
  return __sync_swap_4(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned int
__spirv_AtomicExchange(local unsigned int *p, unsigned int scope,
                       unsigned int semantics, unsigned int val) {
  return __sync_swap_4(p, val);
}

#ifdef cl_khr_int64_extended_atomics
_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicExchange(volatile global long *p,
                                                   unsigned int scope,
                                                   unsigned int semantics,
                                                   long val) {
  return __sync_swap_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicExchange(volatile local long *p,
                                                   unsigned int scope,
                                                   unsigned int semantics,
                                                   long val) {
  return __sync_swap_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicExchange(volatile global unsigned long *p, unsigned int scope,
                       unsigned int semantics, unsigned long val) {
  return __sync_swap_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicExchange(volatile local unsigned long *p, unsigned int scope,
                       unsigned int semantics, unsigned long val) {
  return __sync_swap_8(p, val);
}
_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicExchange(global long *p,
                                                   unsigned int scope,
                                                   unsigned int semantics,
                                                   long val) {
  return __sync_swap_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicExchange(local long *p,
                                                   unsigned int scope,
                                                   unsigned int semantics,
                                                   long val) {
  return __sync_swap_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicExchange(global unsigned long *p, unsigned int scope,
                       unsigned int semantics, unsigned long val) {
  return __sync_swap_8(p, val);
}

_CLC_OVERLOAD _CLC_DEF unsigned long
__spirv_AtomicExchange(local unsigned long *p, unsigned int scope,
                       unsigned int semantics, unsigned long val) {
  return __sync_swap_8(p, val);
}
#endif // cl_khr_int64_extended_atomics
