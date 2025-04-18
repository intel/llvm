//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_DECL int
_Z24__spirv_AtomicIIncrementPU3AS3iN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(
    volatile local int *, enum Scope, enum MemorySemanticsMask);
_CLC_DECL int
_Z24__spirv_AtomicIIncrementPU3AS1iN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(
    volatile global int *, enum Scope, enum MemorySemanticsMask);
_CLC_DECL uint
_Z24__spirv_AtomicIIncrementPU3AS3jN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(
    volatile local uint *, enum Scope, enum MemorySemanticsMask);
_CLC_DECL uint
_Z24__spirv_AtomicIIncrementPU3AS1jN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(
    volatile global uint *, enum Scope, enum MemorySemanticsMask);

#ifdef cl_khr_int64_base_atomics
_CLC_DECL long
_Z24__spirv_AtomicIIncrementPU3AS3lN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(
    volatile local long *, enum Scope, enum MemorySemanticsMask);
_CLC_DECL long
_Z24__spirv_AtomicIIncrementPU3AS1lN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(
    volatile global long *, enum Scope, enum MemorySemanticsMask);
_CLC_DECL unsigned long
_Z24__spirv_AtomicIIncrementPU3AS3mN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(
    volatile local unsigned long *, enum Scope, enum MemorySemanticsMask);
_CLC_DECL unsigned long
_Z24__spirv_AtomicIIncrementPU3AS1mN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(
    volatile global unsigned long *, enum Scope, enum MemorySemanticsMask);
#endif
