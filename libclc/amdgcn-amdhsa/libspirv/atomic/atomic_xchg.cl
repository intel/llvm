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

AMDGPU_ATOMIC(_Z22__spirv_AtomicExchange, int, i, global, AS1,
              __hip_atomic_exchange)
AMDGPU_ATOMIC(_Z22__spirv_AtomicExchange, unsigned int, j, global, AS1,
              __hip_atomic_exchange)
AMDGPU_ATOMIC(_Z22__spirv_AtomicExchange, int, i, local, AS3,
              __hip_atomic_exchange)
AMDGPU_ATOMIC(_Z22__spirv_AtomicExchange, unsigned int, j, local, AS3,
              __hip_atomic_exchange)

AMDGPU_ATOMIC(_Z22__spirv_AtomicExchange, long, l, global, AS1,
              __hip_atomic_exchange)
AMDGPU_ATOMIC(_Z22__spirv_AtomicExchange, unsigned long, m, global, AS1,
              __hip_atomic_exchange)
AMDGPU_ATOMIC(_Z22__spirv_AtomicExchange, long, l, local, AS3,
              __hip_atomic_exchange)
AMDGPU_ATOMIC(_Z22__spirv_AtomicExchange, unsigned long, m, local, AS3,
              __hip_atomic_exchange)

AMDGPU_ATOMIC(_Z22__spirv_AtomicExchange, float, f, global, AS1,
              __hip_atomic_exchange)
AMDGPU_ATOMIC(_Z22__spirv_AtomicExchange, float, f, local, AS3,
              __hip_atomic_exchange)

// TODO implement for fp64

#undef AMDGPU_ATOMIC
