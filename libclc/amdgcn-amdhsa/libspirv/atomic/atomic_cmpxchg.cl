//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// CompareExceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "atomic_helpers.h"
#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

AMDGPU_ATOMIC(_Z22__spirv_AtomicCompareExchange, int, i, global, AS1,
              __atomic_compare_exchange)
AMDGPU_ATOMIC(_Z22__spirv_AtomicCompareExchange, unsigned int, j, global, AS1,
              __atomic_compare_exchange)
AMDGPU_ATOMIC(_Z22__spirv_AtomicCompareExchange, int, i, local, AS3,
              __atomic_compare_exchange)
AMDGPU_ATOMIC(_Z22__spirv_AtomicCompareExchange, unsigned int, j, local, AS3,
              __atomic_compare_exchange)

AMDGPU_ATOMIC(_Z22__spirv_AtomicCompareExchange, long, l, global, AS1,
              __atomic_compare_exchange)
AMDGPU_ATOMIC(_Z22__spirv_AtomicCompareExchange, unsigned long, m, global, AS1,
              __atomic_compare_exchange)
AMDGPU_ATOMIC(_Z22__spirv_AtomicCompareExchange, long, l, local, AS3,
              __atomic_compare_exchange)
AMDGPU_ATOMIC(_Z22__spirv_AtomicCompareExchange, unsigned long, m, local, AS3,
              __atomic_compare_exchange)

AMDGPU_ATOMIC(_Z22__spirv_AtomicCompareExchange, float, f, global, AS1,
              __atomic_compare_exchange)
AMDGPU_ATOMIC(_Z22__spirv_AtomicCompareExchange, float, f, local, AS3,
              __atomic_compare_exchange)

// TODO implement for fp64

#undef AMDGPU_ATOMIC
