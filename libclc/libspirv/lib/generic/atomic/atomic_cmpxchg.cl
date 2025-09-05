//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicCompareExchange(local int *p,
                                                         int scope, int eq,
                                                         int neq, int val,
                                                         int cmp) {
  return __sync_val_compare_and_swap(p, cmp, val);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicCompareExchange(global int *p,
                                                         int scope, int eq,
                                                         int neq, int val,
                                                         int cmp) {
  return __sync_val_compare_and_swap(p, cmp, val);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_AtomicCompareExchange(local uint *p,
                                                          int scope, int eq,
                                                          int neq, uint val,
                                                          uint cmp) {
  return __sync_val_compare_and_swap(p, cmp, val);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_AtomicCompareExchange(global uint *p,
                                                          int scope, int eq,
                                                          int neq, uint val,
                                                          uint cmp) {
  return __sync_val_compare_and_swap(p, cmp, val);
}

#ifdef cl_khr_int64_base_atomics
_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicCompareExchange(local long *p,
                                                          int scope, int eq,
                                                          int neq, long val,
                                                          long cmp) {
  return __sync_val_compare_and_swap_8(p, cmp, val);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicCompareExchange(global long *p,
                                                          int scope, int eq,
                                                          int neq, long val,
                                                          long cmp) {
  return __sync_val_compare_and_swap_8(p, cmp, val);
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_AtomicCompareExchange(local ulong *p,
                                                           int scope, int eq,
                                                           int neq, ulong val,
                                                           ulong cmp) {
  return __sync_val_compare_and_swap_8(p, cmp, val);
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_AtomicCompareExchange(global ulong *p,
                                                           int scope, int eq,
                                                           int neq, ulong val,
                                                           ulong cmp) {
  return __sync_val_compare_and_swap_8(p, cmp, val);
}

#endif

#if _CLC_GENERIC_AS_SUPPORTED

_CLC_OVERLOAD _CLC_DEF int __spirv_AtomicCompareExchange(int *p, int scope,
                                                         int eq, int neq,
                                                         int val, int cmp) {
  return __sync_val_compare_and_swap(p, cmp, val);
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_AtomicCompareExchange(uint *p, int scope,
                                                          int eq, int neq,
                                                          uint val, uint cmp) {
  return __sync_val_compare_and_swap(p, cmp, val);
}

#ifdef cl_khr_int64_base_atomics

_CLC_OVERLOAD _CLC_DEF long __spirv_AtomicCompareExchange(long *p, int scope,
                                                          int eq, int neq,
                                                          long val, long cmp) {
  return __sync_val_compare_and_swap_8(p, cmp, val);
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_AtomicCompareExchange(ulong *p, int scope,
                                                           int eq, int neq,
                                                           ulong val,
                                                           ulong cmp) {
  return __sync_val_compare_and_swap_8(p, cmp, val);
}

#endif

#endif //_CLC_GENERIC_AS_SUPPORTED
