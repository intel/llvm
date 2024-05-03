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

#define __CLC_OP ^
#define __SPIRV_BUILTIN _Z17__spirv_AtomicXor
#define __HIP_BUILTIN __hip_atomic_fetch_xor

#include "atomic_safe.def"
