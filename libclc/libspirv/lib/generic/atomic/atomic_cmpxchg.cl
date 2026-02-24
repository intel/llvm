//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/atomic/clc_atomic_compare_exchange.h>
#include <libspirv/atomic/atomic_helper.h>
#include <libspirv/spirv.h>

#define __CLC_FUNCTION __spirv_AtomicCompareExchange
#define __CLC_IMPL_FUNCTION __clc_atomic_compare_exchange
#define __CLC_COMPARE_EXCHANGE

#define __CLC_BODY <atomic_def.inc>
#include <clc/integer/gentype.inc>

#define __CLC_FLOAT_ONLY
#define __CLC_BODY <atomic_def.inc>
#include <clc/math/gentype.inc>

#undef __CLC_FUNCTION
#define __CLC_FUNCTION __spirv_AtomicCompareExchangeWeak

#define __CLC_BODY <atomic_def.inc>
#include <clc/integer/gentype.inc>
