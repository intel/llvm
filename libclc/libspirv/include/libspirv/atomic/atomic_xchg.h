//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __SPIRV_FUNCTION_S __spirv_AtomicExchange
#define __SPIRV_FUNCTION_S_LEN 22
#define __SPIRV_FUNCTION_U __spirv_AtomicExchange
#define __SPIRV_FUNCTION_U_LEN 22
#define __SPIRV_INT64_BASE

// TODO: Stop manually mangling this name. Need C++ namespaces to get the exact
// mangling.
_CLC_DECL float
_Z22__spirv_AtomicExchangePU3AS3fN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEf(
    volatile local float *, enum Scope, enum MemorySemanticsMask, float);
_CLC_DECL float
_Z22__spirv_AtomicExchangePU3AS1fN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEf(
    volatile global float *, enum Scope, enum MemorySemanticsMask, float);
#include <libspirv/atomic/atomic_decl.inc>
