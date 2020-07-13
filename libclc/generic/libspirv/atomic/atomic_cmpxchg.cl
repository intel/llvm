//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicCompareExchange(volatile local int *p,
                                                         unsigned int scope,
                                                         unsigned int eq,
                                                         unsigned int neq,
                                                         int val, int cmp) {
  return __sync_val_compare_and_swap(p, cmp, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicCompareExchange(volatile global int *p,
                                                         unsigned int scope,
                                                         unsigned int eq,
                                                         unsigned int neq,
                                                         int val, int cmp) {
  return __sync_val_compare_and_swap(p, cmp, val);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_AtomicCompareExchange(
    volatile local uint *p, unsigned int scope, unsigned int eq,
    unsigned int neq, uint val, uint cmp) {
  return __sync_val_compare_and_swap(p, cmp, val);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_AtomicCompareExchange(
    volatile global uint *p, unsigned int scope, unsigned int eq,
    unsigned int neq, uint val, uint cmp) {
  return __sync_val_compare_and_swap(p, cmp, val);
}

_CLC_OVERLOAD _CLC_DEF int
__spirv_AtomicCompareExchange(local int *p, unsigned int scope, unsigned int eq,
                              unsigned int neq, int val, int cmp) {
  return __sync_val_compare_and_swap(p, cmp, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicCompareExchange(global int *p,
                                                         unsigned int scope,
                                                         unsigned int eq,
                                                         unsigned int neq,
                                                         int val, int cmp) {
  return __sync_val_compare_and_swap(p, cmp, val);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_AtomicCompareExchange(local uint *p,
                                                          unsigned int scope,
                                                          unsigned int eq,
                                                          unsigned int neq,
                                                          uint val, uint cmp) {
  return __sync_val_compare_and_swap(p, cmp, val);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_AtomicCompareExchange(global uint *p,
                                                          unsigned int scope,
                                                          unsigned int eq,
                                                          unsigned int neq,
                                                          uint val, uint cmp) {
  return __sync_val_compare_and_swap(p, cmp, val);
}

#ifdef cl_khr_int64_base_atomics
_CLC_OVERLOAD _CLC_DEF long
__spirv_AtomicCompareExchange(volatile local long *p, unsigned int scope,
                              unsigned int eq, unsigned int neq, long val,
                              long cmp) {
  return __sync_val_compare_and_swap_8(p, cmp, val);
}

_CLC_OVERLOAD _CLC_DEF long
__spirv_AtomicCompareExchange(volatile global long *p, unsigned int scope,
                              unsigned int eq, unsigned int neq, long val,
                              long cmp) {
  return __sync_val_compare_and_swap_8(p, cmp, val);
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_AtomicCompareExchange(
    volatile local ulong *p, unsigned int scope, unsigned int eq,
    unsigned int neq, ulong val, ulong cmp) {
  return __sync_val_compare_and_swap_8(p, cmp, val);
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_AtomicCompareExchange(
    volatile global ulong *p, unsigned int scope, unsigned int eq,
    unsigned int neq, ulong val, ulong cmp) {
  return __sync_val_compare_and_swap_8(p, cmp, val);
}
_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicCompareExchange(local long *p,
                                                          unsigned int scope,
                                                          unsigned int eq,
                                                          unsigned int neq,
                                                          long val, long cmp) {
  return __sync_val_compare_and_swap_8(p, cmp, val);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicCompareExchange(global long *p,
                                                          unsigned int scope,
                                                          unsigned int eq,
                                                          unsigned int neq,
                                                          long val, long cmp) {
  return __sync_val_compare_and_swap_8(p, cmp, val);
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_AtomicCompareExchange(
    local ulong *p, unsigned int scope, unsigned int eq, unsigned int neq,
    ulong val, ulong cmp) {
  return __sync_val_compare_and_swap_8(p, cmp, val);
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_AtomicCompareExchange(
    global ulong *p, unsigned int scope, unsigned int eq, unsigned int neq,
    ulong val, ulong cmp) {
  return __sync_val_compare_and_swap_8(p, cmp, val);
}
#endif // cl_khr_int64_base_atomics
