//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "atomic_helpers.h"
#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

AMDGPU_ATOMIC(_Z18__spirv_AtomicSMax, int, i, global, AS1,
              __hip_atomic_fetch_max)
AMDGPU_ATOMIC(_Z18__spirv_AtomicUMax, unsigned int, j, global, AS1,
              __hip_atomic_fetch_max)
AMDGPU_ATOMIC(_Z18__spirv_AtomicSMax, int, i, local, AS3,
              __hip_atomic_fetch_max)
AMDGPU_ATOMIC(_Z18__spirv_AtomicUMax, unsigned int, j, local, AS3,
              __hip_atomic_fetch_max)

#ifdef cl_khr_int64_base_atomics
AMDGPU_ATOMIC(_Z18__spirv_AtomicSMax, long, l, global, AS1,
              __hip_atomic_fetch_max)
AMDGPU_ATOMIC(_Z18__spirv_AtomicUMax, unsigned long, m, global, AS1,
              __hip_atomic_fetch_max)
AMDGPU_ATOMIC(_Z18__spirv_AtomicSMax, long, l, local, AS3,
              __hip_atomic_fetch_max)
AMDGPU_ATOMIC(_Z18__spirv_AtomicUMax, unsigned long, m, local, AS3,
              __hip_atomic_fetch_max)
#endif

/*

   TODO atomic_fetch_max is broken when ptr[0] < 0 and val > 0; 

AMDGPU_ATOMIC(_Z21__spirv_AtomicFMaxEXT, float, f, global, AS1,
              __hip_atomic_fetch_max)
AMDGPU_ATOMIC(_Z21__spirv_AtomicFMaxEXT, float, f, local, AS3,
              __hip_atomic_fetch_max)
*/

// TODO implement for fp64

#undef AMDGPU_ATOMIC
