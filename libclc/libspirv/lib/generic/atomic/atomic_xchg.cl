//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF float __spirv_AtomicExchange(global float *p, int scope,
                                                    int semantics, float val) {
  return __clc_as_float(__spirv_AtomicExchange((global uint *)p, scope,
                                               semantics, __clc_as_uint(val)));
}

_CLC_OVERLOAD _CLC_DEF float __spirv_AtomicExchange(local float *p, int scope,
                                                    int semantics, float val) {
  return __clc_as_float(__spirv_AtomicExchange((local uint *)p, scope,
                                               semantics, __clc_as_uint(val)));
}

#define IMPL(TYPE, AS, FN_NAME)                                                \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_AtomicExchange(                          \
      AS TYPE *p, int scope, int semantics, TYPE val) {                        \
    return FN_NAME(p, val);                                                    \
  }

IMPL(int, global, __sync_swap_4)
IMPL(int, local, __sync_swap_4)

#ifdef cl_khr_int64_base_atomics
IMPL(long, global, __sync_swap_8)
IMPL(long, local, __sync_swap_8)
#endif

#if _CLC_GENERIC_AS_SUPPORTED

#define IMPL_GENERIC(TYPE, FN_NAME) IMPL(TYPE, , FN_NAME)

IMPL_GENERIC(int, __sync_swap_4)

#ifdef cl_khr_int64_base_atomics
IMPL_GENERIC(long, __sync_swap_8)
#endif

#endif //_CLC_GENERIC_AS_SUPPORTED

#undef IMPL
