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

__CLC_NVVM_ATOMIC(int, int, i, and, __spirv_AtomicAnd)
__CLC_NVVM_ATOMIC(long, long, l, and, __spirv_AtomicAnd)
__CLC_NVVM_ATOMIC(unsigned int, int, i, and, __spirv_AtomicAnd)
__CLC_NVVM_ATOMIC(unsigned long, long, l, and, __spirv_AtomicAnd)

#undef __CLC_NVVM_ATOMIC_TYPES
#undef __CLC_NVVM_ATOMIC
#undef __CLC_NVVM_ATOMIC_IMPL
