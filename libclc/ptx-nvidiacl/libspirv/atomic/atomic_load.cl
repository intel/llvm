//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <atomic_helpers.h>
#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

extern int __clc_nvvm_reflect_arch();
_CLC_OVERLOAD _CLC_DECL void __spirv_MemoryBarrier(unsigned int, unsigned int);

#define __CLC_NVVM_ATOMIC_LOAD_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV,      \
                                          ADDR_SPACE, ADDR_SPACE_NV, ORDER)    \
  switch (scope) {                                                             \
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

#define __CLC_NVVM_ATOMIC_LOAD_IMPL(                                           \
    TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV, ADDR_SPACE,                  \
    POINTER_AND_ADDR_SPACE_MANGLED, ADDR_SPACE_NV)                             \
  __attribute__((always_inline)) _CLC_DECL TYPE _Z18__spirv_\
AtomicLoad##POINTER_AND_ADDR_SPACE_MANGLED##K##TYPE_MANGLED##N5__spv5\
Scope4FlagENS1_19MemorySemanticsMask4FlagE(                                    \
      const volatile ADDR_SPACE TYPE *pointer, enum Scope scope,               \
      enum MemorySemanticsMask semantics) {                                    \
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

#define __CLC_NVVM_ATOMIC_LOAD(TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV)   \
  __CLC_NVVM_ATOMIC_LOAD_IMPL(TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV,    \
                              __global, PU3AS1, _global_)                      \
  __CLC_NVVM_ATOMIC_LOAD_IMPL(TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV,    \
                              __local, PU3AS3, _shared_)                       \
  __CLC_NVVM_ATOMIC_LOAD_IMPL(TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV, ,  \
                              P, _gen_)

__CLC_NVVM_ATOMIC_LOAD(int, i, int, i)
__CLC_NVVM_ATOMIC_LOAD(uint, j, int, i)
__CLC_NVVM_ATOMIC_LOAD(long, l, long, l)
__CLC_NVVM_ATOMIC_LOAD(ulong, m, long, l)

__CLC_NVVM_ATOMIC_LOAD(float, f, float, f)
#ifdef cl_khr_int64_base_atomics
__CLC_NVVM_ATOMIC_LOAD(double, d, double, d)
#endif

#undef __CLC_NVVM_ATOMIC_LOAD_TYPES
#undef __CLC_NVVM_ATOMIC_LOAD
#undef __CLC_NVVM_ATOMIC_LOAD_IMPL
