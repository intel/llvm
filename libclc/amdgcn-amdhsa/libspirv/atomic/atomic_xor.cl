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

#define __CLC_XOR ^

AMDGPU_CAS_ATOMIC(_Z17__spirv_AtomicXor, int, i, __CLC_XOR)
AMDGPU_CAS_ATOMIC(_Z17__spirv_AtomicXor, unsigned int, j, __CLC_XOR)
AMDGPU_CAS_ATOMIC(_Z17__spirv_AtomicXor, long, l, __CLC_XOR)
AMDGPU_CAS_ATOMIC(_Z17__spirv_AtomicXor, unsigned long, m, __CLC_XOR)

#undef __CLC_XOR
#undef AMDGPU_ATOMIC
#undef AMDGPU_ATOMIC_IMPL
#undef AMDGPU_ARCH_GEQ
#undef AMDGPU_ARCH_BETWEEN
#undef GET_ATOMIC_SCOPE_AND_ORDER
