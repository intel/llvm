//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: Stop manually mangling this name. Need C++ namespaces to get the exact
// mangling.
_CLC_DECL int
_Z29__spirv_AtomicCompareExchangePU3AS3iN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_ii(
    volatile local int *, enum Scope, enum MemorySemanticsMask,
    enum MemorySemanticsMask, int, int);
_CLC_DECL int
_Z29__spirv_AtomicCompareExchangePU3AS1iN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_ii(
    volatile global int *, enum Scope, enum MemorySemanticsMask,
    enum MemorySemanticsMask, int, int);
_CLC_DECL uint
_Z29__spirv_AtomicCompareExchangePU3AS3jN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_jj(
    volatile local uint *, enum Scope, enum MemorySemanticsMask,
    enum MemorySemanticsMask, uint, uint);
_CLC_DECL uint
_Z29__spirv_AtomicCompareExchangePU3AS1jN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_jj(
    volatile global uint *, enum Scope, enum MemorySemanticsMask,
    enum MemorySemanticsMask, uint, uint);

#ifdef cl_khr_int64_base_atomics
_CLC_DECL long
_Z29__spirv_AtomicCompareExchangePU3AS3lN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_ll(
    volatile local long *, enum Scope, enum MemorySemanticsMask,
    enum MemorySemanticsMask, long, long);
_CLC_DECL long
_Z29__spirv_AtomicCompareExchangePU3AS1lN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_ll(
    volatile global long *, enum Scope, enum MemorySemanticsMask,
    enum MemorySemanticsMask, long, long);
_CLC_DECL unsigned long
_Z29__spirv_AtomicCompareExchangePU3AS3mN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_mm(
    volatile local unsigned long *, enum Scope, enum MemorySemanticsMask,
    enum MemorySemanticsMask, unsigned long, unsigned long);
_CLC_DECL unsigned long
_Z29__spirv_AtomicCompareExchangePU3AS1mN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_mm(
    volatile global unsigned long *, enum Scope, enum MemorySemanticsMask,
    enum MemorySemanticsMask, unsigned long, unsigned long);
#endif
