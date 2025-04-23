//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DECL int __spirv_AtomicIIncrement(local int *, int Scope,
                                                     enum MemorySemanticsMask);
_CLC_OVERLOAD _CLC_DECL int __spirv_AtomicIIncrement(global int *, int Scope,
                                                     enum MemorySemanticsMask);
_CLC_OVERLOAD _CLC_DECL uint __spirv_AtomicIIncrement(local uint *, int Scope,
                                                      enum MemorySemanticsMask);
_CLC_OVERLOAD _CLC_DECL uint __spirv_AtomicIIncrement(global uint *, int Scope,
                                                      enum MemorySemanticsMask);

#ifdef cl_khr_int64_base_atomics
_CLC_OVERLOAD _CLC_DECL long __spirv_AtomicIIncrement(local long *, int Scope,
                                                      enum MemorySemanticsMask);
_CLC_OVERLOAD _CLC_DECL long __spirv_AtomicIIncrement(global long *, int Scope,
                                                      enum MemorySemanticsMask);
_CLC_OVERLOAD _CLC_DECL unsigned long
__spirv_AtomicIIncrement(local unsigned long *, int Scope,
                         enum MemorySemanticsMask);
_CLC_OVERLOAD _CLC_DECL unsigned long
__spirv_AtomicIIncrement(global unsigned long *, int Scope,
                         enum MemorySemanticsMask);
#endif
