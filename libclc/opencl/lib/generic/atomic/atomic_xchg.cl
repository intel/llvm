//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/clc.h>
#include <libspirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF float atomic_xchg(volatile global float *p, float val) {
  return __spirv_AtomicExchange((global float *)p, Device,
                                SequentiallyConsistent, val);
}

_CLC_OVERLOAD _CLC_DEF float atomic_xchg(volatile local float *p, float val) {
  return __spirv_AtomicExchange((local float *)p, Device,
                                SequentiallyConsistent, val);
}

#define IMPL(TYPE, AS)                                                         \
  _CLC_OVERLOAD _CLC_DEF TYPE atomic_xchg(volatile AS TYPE *p, TYPE val) {     \
    return __spirv_AtomicExchange((AS TYPE *)p, Device,                        \
                                  SequentiallyConsistent, val);                \
  }

IMPL(int, global)
IMPL(unsigned int, global)
IMPL(int, local)
IMPL(unsigned int, local)
#undef IMPL
