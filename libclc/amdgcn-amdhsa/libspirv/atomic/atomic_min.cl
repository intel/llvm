//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "atomic_helpers.h"
#include "atomic_minmax.h"
#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

AMDGPU_ATOMIC(_Z18__spirv_AtomicSMin, int, i, __hip_atomic_fetch_min)
AMDGPU_ATOMIC(_Z18__spirv_AtomicUMin, unsigned int, j, __hip_atomic_fetch_min)
AMDGPU_ATOMIC(_Z18__spirv_AtomicSMin, long, l, __hip_atomic_fetch_min)
AMDGPU_ATOMIC(_Z18__spirv_AtomicUMin, unsigned long, m, __hip_atomic_fetch_min)

AMDGPU_ATOMIC_FP_MINMAX_IMPL(Min, <, float, f, int, i, global, U3AS1, 1, 5_ii)
AMDGPU_ATOMIC_FP_MINMAX_IMPL(Min, <, float, f, int, i, local, U3AS3, 1, 5_ii)
AMDGPU_ATOMIC_FP_MINMAX_IMPL(Min, <, float, f, int, i, , , 0, 4_ii)

#ifdef cl_khr_int64_base_atomics
AMDGPU_ATOMIC_FP_MINMAX_IMPL(Min, <, double, d, long, l, global, U3AS1, 1, 5_ll)
AMDGPU_ATOMIC_FP_MINMAX_IMPL(Min, <, double, d, long, l, local, U3AS3, 1, 5_ll)
AMDGPU_ATOMIC_FP_MINMAX_IMPL(Min, <, double, d, long, l, , , 0, 4_ll)
#endif

#undef AMDGPU_ATOMIC
#undef AMDGPU_ATOMIC_IMPL
#undef AMDGPU_ATOMIC_FP_MINMAX_IMPL
#undef GET_ATOMIC_SCOPE_AND_ORDER
