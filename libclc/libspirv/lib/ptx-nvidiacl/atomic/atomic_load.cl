//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <atomic_helpers.h>
#include <libspirv/spirv.h>
#include <libspirv/spirv_types.h>

extern int __clc_nvvm_reflect_arch();
_CLC_OVERLOAD _CLC_DECL void __spirv_MemoryBarrier(int, int);

#define __CLC_NVVM_ATOMIC_LOAD_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV,      \
                                          ADDR_SPACE, ADDR_SPACE_NV, ORDER)    \
  switch (scope) {                                                             \
  case Invocation:                                                             \
  case Subgroup:                                                               \
  case Workgroup: {                                                            \
    TYPE_NV res = __nvvm##ORDER##_cta_ld##ADDR_SPACE_NV##TYPE_MANGLED_NV(      \
        (ADDR_SPACE TYPE_NV *)pointer);                                        \
    return *(TYPE *)&res;                                                      \
  }                                                                            \
  case Device: {                                                               \
    TYPE_NV res = __nvvm##ORDER##_ld##ADDR_SPACE_NV##TYPE_MANGLED_NV(          \
        (ADDR_SPACE TYPE_NV *)pointer);                                        \
    return *(TYPE *)&res;                                                      \
  }                                                                            \
  case CrossDevice:                                                            \
  default: {                                                                   \
    TYPE_NV res = __nvvm##ORDER##_sys_ld##ADDR_SPACE_NV##TYPE_MANGLED_NV(      \
        (ADDR_SPACE TYPE_NV *)pointer);                                        \
    return *(TYPE *)&res;                                                      \
  }                                                                            \
  }

#define __CLC_NVVM_ATOMIC_LOAD_IMPL(TYPE, TYPE_NV, TYPE_MANGLED_NV,            \
                                    ADDR_SPACE, ADDR_SPACE_NV)                 \
  __attribute__((always_inline)) _CLC_OVERLOAD _CLC_DECL TYPE                  \
  __spirv_AtomicLoad(ADDR_SPACE TYPE *pointer, int scope, int semantics) {     \
    /* Semantics mask may include memory order, storage class and other info   \
Memory order is stored in the lowest 5 bits */                                 \
    unsigned int order = semantics & 0x1F;                                     \
    if (__clc_nvvm_reflect_arch() >= 700) {                                    \
      switch (order) {                                                         \
      case None:                                                               \
        __CLC_NVVM_ATOMIC_LOAD_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV,      \
                                          ADDR_SPACE, ADDR_SPACE_NV, )         \
      case Acquire:                                                            \
        __CLC_NVVM_ATOMIC_LOAD_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV,      \
                                          ADDR_SPACE, ADDR_SPACE_NV, _acquire) \
        break;                                                                 \
      case SequentiallyConsistent:                                             \
        __CLC_NVVM_FENCE_SC_SM70()                                             \
        __CLC_NVVM_ATOMIC_LOAD_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV,      \
                                          ADDR_SPACE, ADDR_SPACE_NV, _acquire) \
        break;                                                                 \
      }                                                                        \
    } else {                                                                   \
      TYPE_NV res = __nvvm_volatile_ld##ADDR_SPACE_NV##TYPE_MANGLED_NV(        \
          (ADDR_SPACE TYPE_NV *)pointer);                                      \
      switch (order) {                                                         \
      case None:                                                               \
        return *(TYPE *)&res;                                                  \
      case Acquire: {                                                          \
        __spirv_MemoryBarrier(scope, Acquire);                                 \
        return *(TYPE *)&res;                                                  \
      }                                                                        \
      }                                                                        \
    }                                                                          \
    __builtin_trap();                                                          \
    __builtin_unreachable();                                                   \
  }

#define __CLC_NVVM_ATOMIC_LOAD(TYPE, TYPE_NV, TYPE_MANGLED_NV)                 \
  __CLC_NVVM_ATOMIC_LOAD_IMPL(TYPE, TYPE_NV, TYPE_MANGLED_NV, __global,        \
                              _global_)                                        \
  __CLC_NVVM_ATOMIC_LOAD_IMPL(TYPE, TYPE_NV, TYPE_MANGLED_NV, __local,         \
                              _shared_)                                        \
  __CLC_NVVM_ATOMIC_LOAD_IMPL(TYPE, TYPE_NV, TYPE_MANGLED_NV, , _gen_)

__CLC_NVVM_ATOMIC_LOAD(int, int, i)
__CLC_NVVM_ATOMIC_LOAD(long, long, l)

__CLC_NVVM_ATOMIC_LOAD(float, float, f)
#ifdef cl_khr_int64_base_atomics
__CLC_NVVM_ATOMIC_LOAD(double, double, d)
#endif

#undef __CLC_NVVM_ATOMIC_LOAD_TYPES
#undef __CLC_NVVM_ATOMIC_LOAD
#undef __CLC_NVVM_ATOMIC_LOAD_IMPL
