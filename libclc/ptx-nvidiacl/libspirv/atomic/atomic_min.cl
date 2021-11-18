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

__CLC_NVVM_ATOMIC(int, i, int, i, min, _Z18__spirv_AtomicSMin)
__CLC_NVVM_ATOMIC(long, l, long, l, min, _Z18__spirv_AtomicSMin)
__CLC_NVVM_ATOMIC(uint, j, uint, ui, min, _Z18__spirv_AtomicUMin)
__CLC_NVVM_ATOMIC(ulong, m, ulong, ul, min, _Z18__spirv_AtomicUMin)

#undef __CLC_NVVM_ATOMIC_TYPES
#undef __CLC_NVVM_ATOMIC
#undef __CLC_NVVM_ATOMIC_IMPL
