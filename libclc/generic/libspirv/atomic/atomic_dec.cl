//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicIDecrement(const local int *p,
                                                    unsigned int scope,
                                                    unsigned int semantics) {
  return __sync_fetch_and_sub((local int *)p, (int)1);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicIDecrement(const global int *p,
                                                    unsigned int scope,
                                                    unsigned int semantics) {
  return __sync_fetch_and_sub((global int *)p, (int)1);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_AtomicIDecrement(const local uint *p,
                                                     unsigned int scope,
                                                     unsigned int semantics) {
  return __sync_fetch_and_sub((local uint *)p, (uint)1);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_AtomicIDecrement(const global uint *p,
                                                     unsigned int scope,
                                                     unsigned int semantics) {
  return __sync_fetch_and_sub((global uint *)p, (uint)1);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicIDecrement(const volatile local int *p,
                                                    unsigned int scope,
                                                    unsigned int semantics) {
  return __sync_fetch_and_sub((volatile local int *)p, (int)1);
}

_CLC_OVERLOAD _CLC_DEF int
__spirv_AtomicIDecrement(const volatile global int *p, unsigned int scope,
                         unsigned int semantics) {
  return __sync_fetch_and_sub((volatile global int *)p, (int)1);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_AtomicIDecrement(
    const volatile local uint *p, unsigned int scope, unsigned int semantics) {
  return __sync_fetch_and_sub((volatile local uint *)p, (uint)1);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_AtomicIDecrement(
    const volatile global uint *p, unsigned int scope, unsigned int semantics) {
  return __sync_fetch_and_sub((volatile global uint *)p, (uint)1);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicIDecrement(volatile local int *p,
                                                    unsigned int scope,
                                                    unsigned int semantics) {
  return __sync_fetch_and_sub(p, (int)1);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicIDecrement(volatile global int *p,
                                                    unsigned int scope,
                                                    unsigned int semantics) {
  return __sync_fetch_and_sub(p, (int)1);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_AtomicIDecrement(volatile local uint *p,
                                                     unsigned int scope,
                                                     unsigned int semantics) {
  return __sync_fetch_and_sub(p, (uint)1);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_AtomicIDecrement(volatile global uint *p,
                                                     unsigned int scope,
                                                     unsigned int semantics) {
  return __sync_fetch_and_sub(p, (uint)1);
}

#ifdef cl_khr_int64_base_atomics
_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicIDecrement(const local long *p,
                                                     unsigned int scope,
                                                     unsigned int semantics) {
  return __sync_fetch_and_sub((local long *)p, (long)1);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicIDecrement(const global long *p,
                                                     unsigned int scope,
                                                     unsigned int semantics) {
  return __sync_fetch_and_sub((global long *)p, (long)1);
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_AtomicIDecrement(const local ulong *p,
                                                      unsigned int scope,
                                                      unsigned int semantics) {
  return __sync_fetch_and_sub((local ulong *)p, (ulong)1);
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_AtomicIDecrement(const global ulong *p,
                                                      unsigned int scope,
                                                      unsigned int semantics) {
  return __sync_fetch_and_sub((global ulong *)p, (ulong)1);
}

_CLC_OVERLOAD _CLC_DEF long
__spirv_AtomicIDecrement(const volatile local long *p, unsigned int scope,
                         unsigned int semantics) {
  return __sync_fetch_and_sub((volatile local long *)p, (long)1);
}

_CLC_OVERLOAD _CLC_DEF long
__spirv_AtomicIDecrement(const volatile global long *p, unsigned int scope,
                         unsigned int semantics) {
  return __sync_fetch_and_sub((volatile global long *)p, (long)1);
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_AtomicIDecrement(
    const volatile local ulong *p, unsigned int scope, unsigned int semantics) {
  return __sync_fetch_and_sub((volatile local ulong *)p, (ulong)1);
}

_CLC_OVERLOAD _CLC_DEF ulong
__spirv_AtomicIDecrement(const volatile global ulong *p, unsigned int scope,
                         unsigned int semantics) {
  return __sync_fetch_and_sub((volatile global ulong *)p, (ulong)1);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicIDecrement(volatile local long *p,
                                                     unsigned int scope,
                                                     unsigned int semantics) {
  return __sync_fetch_and_sub(p, (long)1);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicIDecrement(volatile global long *p,
                                                     unsigned int scope,
                                                     unsigned int semantics) {
  return __sync_fetch_and_sub(p, (long)1);
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_AtomicIDecrement(volatile local ulong *p,
                                                      unsigned int scope,
                                                      unsigned int semantics) {
  return __sync_fetch_and_sub(p, (ulong)1);
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_AtomicIDecrement(volatile global ulong *p,
                                                      unsigned int scope,
                                                      unsigned int semantics) {
  return __sync_fetch_and_sub(p, (ulong)1);
}
#endif // cl_khr_int64_base_atomics
