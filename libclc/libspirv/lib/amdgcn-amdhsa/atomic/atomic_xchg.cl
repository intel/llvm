//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "atomic_helpers.h"
#include <libspirv/spirv.h>
#include <libspirv/spirv_types.h>

AMDGPU_ATOMIC(_Z22__spirv_AtomicExchange, int, i, __hip_atomic_exchange)
AMDGPU_ATOMIC(_Z22__spirv_AtomicExchange, unsigned int, j,
              __hip_atomic_exchange)
AMDGPU_ATOMIC(_Z22__spirv_AtomicExchange, long, l, __hip_atomic_exchange)
AMDGPU_ATOMIC(_Z22__spirv_AtomicExchange, unsigned long, m,
              __hip_atomic_exchange)
AMDGPU_ATOMIC(_Z22__spirv_AtomicExchange, float, f, __hip_atomic_exchange)

// TODO implement for fp64

#undef AMDGPU_ATOMIC
#undef AMDGPU_ATOMIC_IMPL
#undef AMDGPU_ARCH_GEQ
#undef AMDGPU_ARCH_BETWEEN
#undef GET_ATOMIC_SCOPE_AND_ORDER
