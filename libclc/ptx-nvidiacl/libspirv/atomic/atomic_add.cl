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

__CLC_NVVM_ATOMIC(int, i, int, i, add, _Z18__spirv_AtomicIAdd)
__CLC_NVVM_ATOMIC(uint, j, int, i, add, _Z18__spirv_AtomicIAdd)
__CLC_NVVM_ATOMIC(long, l, long, l, add, _Z18__spirv_AtomicIAdd)
__CLC_NVVM_ATOMIC(ulong, m, long, l, add, _Z18__spirv_AtomicIAdd)

__CLC_NVVM_ATOMIC(float, f, float, f, add, _Z21__spirv_AtomicFAddEXT)
#ifdef cl_khr_int64_base_atomics
__CLC_NVVM_ATOMIC(double, d, double, d, add, _Z21__spirv_AtomicFAddEXT)
#endif

#undef __CLC_NVVM_ATOMIC_TYPES
#undef __CLC_NVVM_ATOMIC
#undef __CLC_NVVM_ATOMIC_IMPL
