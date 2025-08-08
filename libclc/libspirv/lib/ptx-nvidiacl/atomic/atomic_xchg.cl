//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <atomic_helpers.h>
#include <libspirv/spirv.h>
#include <libspirv/spirv_types.h>

__CLC_NVVM_ATOMIC(int, int, i, xchg, __spirv_AtomicExchange)
__CLC_NVVM_ATOMIC(long, long, l, xchg, __spirv_AtomicExchange)
__CLC_NVVM_ATOMIC(unsigned int, int, i, xchg, __spirv_AtomicExchange)
__CLC_NVVM_ATOMIC(unsigned long, long, l, xchg, __spirv_AtomicExchange)
__CLC_NVVM_ATOMIC(float, float, f, xchg, __spirv_AtomicExchange)
__CLC_NVVM_ATOMIC(double, double, d, xchg, __spirv_AtomicExchange)

#undef __CLC_NVVM_ATOMIC_TYPES
#undef __CLC_NVVM_ATOMIC
#undef __CLC_NVVM_ATOMIC_IMPL
