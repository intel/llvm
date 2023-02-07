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

AMDGPU_ATOMIC(_Z17__spirv_AtomicAnd, int, i, __hip_atomic_fetch_and)
AMDGPU_ATOMIC(_Z17__spirv_AtomicAnd, unsigned int, j, __hip_atomic_fetch_and)
AMDGPU_ATOMIC(_Z17__spirv_AtomicAnd, long, l, __hip_atomic_fetch_and)
AMDGPU_ATOMIC(_Z17__spirv_AtomicAnd, unsigned long, m, __hip_atomic_fetch_and)

#undef AMDGPU_ATOMIC
#undef AMDGPU_ATOMIC_IMPL
#undef GET_ATOMIC_SCOPE_AND_ORDER
