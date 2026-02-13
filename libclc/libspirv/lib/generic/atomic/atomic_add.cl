//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/atomic/clc_atomic_fetch_add.h>
#include <libspirv/atomic/atomic_helper.h>
#include <libspirv/spirv.h>

#define __CLC_FUNCTION __spirv_AtomicIAdd
#define __CLC_IMPL_FUNCTION __clc_atomic_fetch_add

#define __CLC_BODY <atomic_def.inc>
#include <clc/integer/gentype.inc>

#undef __CLC_FUNCTION
#define __CLC_FUNCTION __spirv_AtomicFAddEXT

#define __CLC_BODY <atomic_def.inc>
#include <clc/math/gentype.inc>
