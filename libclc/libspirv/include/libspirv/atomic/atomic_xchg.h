//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __SPIRV_FUNCTION_S __spirv_AtomicExchange
#define __SPIRV_FUNCTION_U __spirv_AtomicExchange
#define __SPIRV_INT64_BASE

_CLC_OVERLOAD _CLC_DECL float __spirv_AtomicExchange(local float *, int Scope,
                                                     int MemorySemanticsMask,
                                                     float);
_CLC_OVERLOAD _CLC_DECL float __spirv_AtomicExchange(global float *, int Scope,
                                                     int MemorySemanticsMask,
                                                     float);
#include <libspirv/atomic/atomic_decl.inc>
