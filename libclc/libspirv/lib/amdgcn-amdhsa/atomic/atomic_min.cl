//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "atomic_helpers.h"
#include "atomic_minmax.h"
#include <libspirv/spirv.h>
#include <libspirv/spirv_types.h>

extern constant int __oclc_ISA_version;

AMDGPU_ATOMIC(__spirv_AtomicSMin, int, __hip_atomic_fetch_min)
AMDGPU_ATOMIC(__spirv_AtomicUMin, unsigned int, __hip_atomic_fetch_min)
AMDGPU_ATOMIC(__spirv_AtomicSMin, long, __hip_atomic_fetch_min)
AMDGPU_ATOMIC(__spirv_AtomicUMin, unsigned long, __hip_atomic_fetch_min)

AMDGPU_ATOMIC_FP_MINMAX_IMPL(Min, <, float, int, global)
AMDGPU_ATOMIC_FP_MINMAX_IMPL(Min, <, float, int, local)
AMDGPU_ATOMIC_FP_MINMAX_IMPL(Min, <, float, int, )

#ifdef cl_khr_int64_base_atomics
AMDGPU_ATOMIC_FP_MINMAX_IMPL_CHECK(Min, <, double, long, global,
                                   AMDGPU_ARCH_BETWEEN(9010, 10000),
                                   __builtin_amdgcn_global_atomic_fmin_f64)
AMDGPU_ATOMIC_FP_MINMAX_IMPL(Min, <, double, long, local)
AMDGPU_ATOMIC_FP_MINMAX_IMPL_CHECK(Min, <, double, long, ,
                                   AMDGPU_ARCH_BETWEEN(9010, 10000),
                                   __builtin_amdgcn_flat_atomic_fmin_f64)
#endif

#undef AMDGPU_ATOMIC
#undef AMDGPU_ATOMIC_IMPL
#undef AMDGPU_ARCH_GEQ
#undef AMDGPU_ARCH_BETWEEN
#undef AMDGPU_ATOMIC_FP_MINMAX_IMPL
#undef GET_ATOMIC_SCOPE_AND_ORDER
