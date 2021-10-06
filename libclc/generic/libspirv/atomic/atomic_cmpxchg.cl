//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

// TODO: Stop manually mangling this name. Need C++ namespaces to get the exact mangling.

_CLC_DEF int
_Z29__spirv_AtomicCompareExchangePU3AS3iN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_ii(
    volatile local int *p, enum Scope scope, enum MemorySemanticsMask eq,
    enum MemorySemanticsMask neq, int val, int cmp) {
  return __sync_val_compare_and_swap(p, cmp, val);
}

_CLC_DEF int
_Z29__spirv_AtomicCompareExchangePU3AS1iN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_ii(
    volatile global int *p, enum Scope scope, enum MemorySemanticsMask eq,
    enum MemorySemanticsMask neq, int val, int cmp) {
  return __sync_val_compare_and_swap(p, cmp, val);
}

_CLC_DEF uint
_Z29__spirv_AtomicCompareExchangePU3AS3jN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_jj(
    volatile local uint *p, enum Scope scope, enum MemorySemanticsMask eq,
    enum MemorySemanticsMask neq, uint val, uint cmp) {
  return __sync_val_compare_and_swap(p, cmp, val);
}

_CLC_DEF uint
_Z29__spirv_AtomicCompareExchangePU3AS1jN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_jj(
    volatile global uint *p, enum Scope scope, enum MemorySemanticsMask eq,
    enum MemorySemanticsMask neq, uint val, uint cmp) {
  return __sync_val_compare_and_swap(p, cmp, val);
}

#ifdef cl_khr_int64_base_atomics
_CLC_DEF long
_Z29__spirv_AtomicCompareExchangePU3AS3lN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_ll(
    volatile local long *p, enum Scope scope, enum MemorySemanticsMask eq,
    enum MemorySemanticsMask neq, long val, long cmp) {
  return __sync_val_compare_and_swap_8(p, cmp, val);
}

_CLC_DEF long
_Z29__spirv_AtomicCompareExchangePU3AS1lN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_ll(
    volatile global long *p, enum Scope scope, enum MemorySemanticsMask eq,
    enum MemorySemanticsMask neq, long val, long cmp) {
  return __sync_val_compare_and_swap_8(p, cmp, val);
}

_CLC_DEF ulong
_Z29__spirv_AtomicCompareExchangePU3AS3mN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_mm(
    volatile local ulong *p, enum Scope scope, enum MemorySemanticsMask eq,
    enum MemorySemanticsMask neq, ulong val, ulong cmp) {
  return __sync_val_compare_and_swap_8(p, cmp, val);
}

_CLC_DEF ulong
_Z29__spirv_AtomicCompareExchangePU3AS1mN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_mm(
    volatile global ulong *p, enum Scope scope, enum MemorySemanticsMask eq,
    enum MemorySemanticsMask neq, ulong val, ulong cmp) {
  return __sync_val_compare_and_swap_8(p, cmp, val);
}
#endif
