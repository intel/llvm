//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/atomic/clc_atomic_fetch_min.h>
#include <libspirv/atomic/atomic_helper.h>
#include <libspirv/spirv.h>

#define __CLC_IMPL_FUNCTION __clc_atomic_fetch_min
#define __CLC_ATOMIC_MIN

#define __CLC_BODY <atomic_minmax.inc>
#include <clc/integer/gentype.inc>

#define __CLC_FUNCTION __spirv_AtomicFMinEXT

#define __CLC_BODY <atomic_def.inc>
#include <clc/math/gentype.inc>
