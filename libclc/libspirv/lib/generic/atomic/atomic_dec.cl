//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_OVERLOAD _CLC_DECL int __spirv_AtomicIDecrement(local int *p, int scope,
                                                     int semantics) {
  return __sync_fetch_and_sub(p, (int)1);
}

_CLC_OVERLOAD _CLC_DECL int __spirv_AtomicIDecrement(global int *p, int scope,
                                                     int semantics) {
  return __sync_fetch_and_sub(p, (int)1);
}

#ifdef cl_khr_int64_base_atomics
_CLC_OVERLOAD _CLC_DECL long __spirv_AtomicIDecrement(local long *p, int scope,
                                                      int semantics) {
  return __sync_fetch_and_sub(p, (long)1);
}

_CLC_OVERLOAD _CLC_DECL long __spirv_AtomicIDecrement(global long *p, int scope,
                                                      int semantics) {
  return __sync_fetch_and_sub(p, (long)1);
}
#endif
