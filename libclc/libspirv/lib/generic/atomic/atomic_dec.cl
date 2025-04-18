//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

// TODO: Stop manually mangling this name. Need C++ namespaces to get the exact mangling.

_CLC_DEF int
_Z24__spirv_AtomicIDecrementPU3AS3iN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(
    volatile local int *p, enum Scope scope,
    enum MemorySemanticsMask semantics) {
  return __sync_fetch_and_sub(p, (int)1);
}

_CLC_DEF int
_Z24__spirv_AtomicIDecrementPU3AS1iN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(
    volatile global int *p, enum Scope scope,
    enum MemorySemanticsMask semantics) {
  return __sync_fetch_and_sub(p, (int)1);
}

_CLC_DEF uint
_Z24__spirv_AtomicIDecrementPU3AS3jN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(
    volatile local uint *p, enum Scope scope,
    enum MemorySemanticsMask semantics) {
  return __sync_fetch_and_sub(p, (uint)1);
}

_CLC_DEF uint
_Z24__spirv_AtomicIDecrementPU3AS1jN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(
    volatile global uint *p, enum Scope scope,
    enum MemorySemanticsMask semantics) {
  return __sync_fetch_and_sub(p, (uint)1);
}

#ifdef cl_khr_int64_base_atomics
_CLC_DEF long
_Z24__spirv_AtomicIDecrementPU3AS3lN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(
    volatile local long *p, enum Scope scope,
    enum MemorySemanticsMask semantics) {
  return __sync_fetch_and_sub(p, (long)1);
}

_CLC_DEF long
_Z24__spirv_AtomicIDecrementPU3AS1lN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(
    volatile global long *p, enum Scope scope,
    enum MemorySemanticsMask semantics) {
  return __sync_fetch_and_sub(p, (long)1);
}

_CLC_DEF ulong
_Z24__spirv_AtomicIDecrementPU3AS3mN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(
    volatile local ulong *p, enum Scope scope,
    enum MemorySemanticsMask semantics) {
  return __sync_fetch_and_sub(p, (ulong)1);
}

_CLC_DEF ulong
_Z24__spirv_AtomicIDecrementPU3AS1mN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(
    volatile global ulong *p, enum Scope scope,
    enum MemorySemanticsMask semantics) {
  return __sync_fetch_and_sub(p, (ulong)1);
}
#endif
