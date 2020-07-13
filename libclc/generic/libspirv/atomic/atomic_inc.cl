//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicIIncrement(const local int *p,
                                                    unsigned int scope,
                                                    unsigned int semantics) {
  return __sync_fetch_and_add((local int *)p, (int)1);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicIIncrement(const global int *p,
                                                    unsigned int scope,
                                                    unsigned int semantics) {
  return __sync_fetch_and_add((global int *)p, (int)1);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_AtomicIIncrement(const local uint *p,
                                                     unsigned int scope,
                                                     unsigned int semantics) {
  return __sync_fetch_and_add((local uint *)p, (uint)1);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_AtomicIIncrement(const global uint *p,
                                                     unsigned int scope,
                                                     unsigned int semantics) {
  return __sync_fetch_and_add((global uint *)p, (uint)1);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicIIncrement(const volatile local int *p,
                                                    unsigned int scope,
                                                    unsigned int semantics) {
  return __sync_fetch_and_add((volatile local int *)p, (int)1);
}

_CLC_OVERLOAD _CLC_DEF int
__spirv_AtomicIIncrement(const volatile global int *p, unsigned int scope,
                         unsigned int semantics) {
  return __sync_fetch_and_add((volatile global int *)p, (int)1);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_AtomicIIncrement(
    const volatile local uint *p, unsigned int scope, unsigned int semantics) {
  return __sync_fetch_and_add((volatile local uint *)p, (uint)1);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_AtomicIIncrement(
    const volatile global uint *p, unsigned int scope, unsigned int semantics) {
  return __sync_fetch_and_add((volatile global uint *)p, (uint)1);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicIIncrement(volatile local int *p,
                                                    unsigned int scope,
                                                    unsigned int semantics) {
  return __sync_fetch_and_add(p, (int)1);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicIIncrement(volatile global int *p,
                                                    unsigned int scope,
                                                    unsigned int semantics) {
  return __sync_fetch_and_add(p, (int)1);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_AtomicIIncrement(volatile local uint *p,
                                                     unsigned int scope,
                                                     unsigned int semantics) {
  return __sync_fetch_and_add(p, (uint)1);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_AtomicIIncrement(volatile global uint *p,
                                                     unsigned int scope,
                                                     unsigned int semantics) {
  return __sync_fetch_and_add(p, (uint)1);
}

#ifdef cl_khr_int64_base_atomics
_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicIIncrement(const local long *p,
                                                     unsigned int scope,
                                                     unsigned int semantics) {
  return __sync_fetch_and_add((local long *)p, (long)1);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicIIncrement(const global long *p,
                                                     unsigned int scope,
                                                     unsigned int semantics) {
  return __sync_fetch_and_add((global long *)p, (long)1);
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_AtomicIIncrement(const local ulong *p,
                                                      unsigned int scope,
                                                      unsigned int semantics) {
  return __sync_fetch_and_add((local ulong *)p, (ulong)1);
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_AtomicIIncrement(const global ulong *p,
                                                      unsigned int scope,
                                                      unsigned int semantics) {
  return __sync_fetch_and_add((global ulong *)p, (ulong)1);
}

_CLC_OVERLOAD _CLC_DEF long
__spirv_AtomicIIncrement(const volatile local long *p, unsigned int scope,
                         unsigned int semantics) {
  return __sync_fetch_and_add((volatile local long *)p, (long)1);
}

_CLC_OVERLOAD _CLC_DEF long
__spirv_AtomicIIncrement(const volatile global long *p, unsigned int scope,
                         unsigned int semantics) {
  return __sync_fetch_and_add((volatile global long *)p, (long)1);
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_AtomicIIncrement(
    const volatile local ulong *p, unsigned int scope, unsigned int semantics) {
  return __sync_fetch_and_add((volatile local ulong *)p, (ulong)1);
}

_CLC_OVERLOAD _CLC_DEF ulong
__spirv_AtomicIIncrement(const volatile global ulong *p, unsigned int scope,
                         unsigned int semantics) {
  return __sync_fetch_and_add((volatile global ulong *)p, (ulong)1);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicIIncrement(volatile local long *p,
                                                     unsigned int scope,
                                                     unsigned int semantics) {
  return __sync_fetch_and_add(p, (long)1);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicIIncrement(volatile global long *p,
                                                     unsigned int scope,
                                                     unsigned int semantics) {
  return __sync_fetch_and_add(p, (long)1);
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_AtomicIIncrement(volatile local ulong *p,
                                                      unsigned int scope,
                                                      unsigned int semantics) {
  return __sync_fetch_and_add(p, (ulong)1);
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_AtomicIIncrement(volatile global ulong *p,
                                                      unsigned int scope,
                                                      unsigned int semantics) {
  return __sync_fetch_and_add(p, (ulong)1);
}
#endif // cl_khr_int64_base_atomics
