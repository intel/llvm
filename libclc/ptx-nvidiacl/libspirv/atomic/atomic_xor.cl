//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <atomic_helpers.h>
#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

__CLC_NVVM_ATOMIC(int, i, int, i, xor, _Z17__spirv_AtomicXor)
__CLC_NVVM_ATOMIC(long, l, long, l, xor, _Z17__spirv_AtomicXor)
__CLC_NVVM_ATOMIC(unsigned int, j, int, i, xor, _Z17__spirv_AtomicXor)
__CLC_NVVM_ATOMIC(unsigned long, m, long, l, xor, _Z17__spirv_AtomicXor)

#undef __CLC_NVVM_ATOMIC_TYPES
#undef __CLC_NVVM_ATOMIC
#undef __CLC_NVVM_ATOMIC_IMPL
