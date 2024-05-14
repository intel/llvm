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

#define __CLC_NVVM_ATOMIC_STORE_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV,     \
                                           ADDR_SPACE, ADDR_SPACE_NV, ORDER)   \
  switch (scope) {                                                             \
  case Subgroup:                                                               \
  case Workgroup: {                                                            \
    __nvvm##ORDER##_cta_st##ADDR_SPACE_NV##TYPE_MANGLED_NV(                    \
        (ADDR_SPACE TYPE_NV *)pointer, *(TYPE_NV *)&value);                    \
    return;                                                                    \
  }                                                                            \
  case Device: {                                                               \
    __nvvm##ORDER##_st##ADDR_SPACE_NV##TYPE_MANGLED_NV(                        \
        (ADDR_SPACE TYPE_NV *)pointer, *(TYPE_NV *)&value);                    \
    return;                                                                    \
  }                                                                            \
  case CrossDevice:                                                            \
  default: {                                                                   \
    __nvvm##ORDER##_sys_st##ADDR_SPACE_NV##TYPE_MANGLED_NV(                    \
        (ADDR_SPACE TYPE_NV *)pointer, *(TYPE_NV *)&value);                    \
    return;                                                                    \
  }                                                                            \
  }

#define __CLC_NVVM_ATOMIC_STORE_IMPL(                                          \
    TYPE, TYPE_MANGLED, SUBSTITUTION, TYPE_NV, TYPE_MANGLED_NV, ADDR_SPACE,    \
    POINTER_AND_ADDR_SPACE_MANGLED, ADDR_SPACE_NV)                             \
  __attribute__((always_inline)) _CLC_DECL void _Z19__spirv_\
AtomicStore##POINTER_AND_ADDR_SPACE_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagEN\
S##SUBSTITUTION##_19MemorySemanticsMask4FlagE##TYPE_MANGLED(                   \
      volatile ADDR_SPACE TYPE *pointer, enum Scope scope,                     \
      enum MemorySemanticsMask semantics, TYPE value) {                        \
    /* Semantics mask may include memory order, storage class and other info   \
Memory order is stored in the lowest 5 bits */                                 \
    unsigned int order = semantics & 0x1F;                                     \
    if (__clc_nvvm_reflect_arch() >= 700) {                                    \
      switch (order) {                                                         \
      case None:                                                               \
        __CLC_NVVM_ATOMIC_STORE_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV,     \
                                           ADDR_SPACE, ADDR_SPACE_NV, )        \
      case Release:                                                            \
        __CLC_NVVM_ATOMIC_STORE_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV,     \
                                           ADDR_SPACE, ADDR_SPACE_NV,          \
                                           _release)                           \
        break;                                                                 \
      case SequentiallyConsistent:                                             \
        __CLC_NVVM_FENCE_SC_SM70()                                             \
        __CLC_NVVM_ATOMIC_STORE_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV,     \
                                           ADDR_SPACE, ADDR_SPACE_NV,          \
                                           _release)                           \
        break;                                                                 \
      }                                                                        \
    } else {                                                                   \
      switch (order) {                                                         \
      case Release:                                                            \
        __spirv_MemoryBarrier(scope, Release);                                 \
        __nvvm_volatile_st##ADDR_SPACE_NV##TYPE_MANGLED_NV(                    \
            (ADDR_SPACE TYPE_NV *)pointer, *(TYPE_NV *)&value);                \
        return;                                                                \
      case None: {                                                             \
        __nvvm_volatile_st##ADDR_SPACE_NV##TYPE_MANGLED_NV(                    \
            (ADDR_SPACE TYPE_NV *)pointer, *(TYPE_NV *)&value);                \
        return;                                                                \
      }                                                                        \
      }                                                                        \
    }                                                                          \
    __builtin_trap();                                                          \
    __builtin_unreachable();                                                   \
  }

#define __CLC_NVVM_ATOMIC_STORE(TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV)  \
  __CLC_NVVM_ATOMIC_STORE_IMPL(TYPE, TYPE_MANGLED, 1, TYPE_NV, TYPE_MANGLED_NV,\
                               __global, PU3AS1, _global_)                     \
  __CLC_NVVM_ATOMIC_STORE_IMPL(TYPE, TYPE_MANGLED, 1, TYPE_NV, TYPE_MANGLED_NV,\
                               __local, PU3AS3, _shared_)                      \
  __CLC_NVVM_ATOMIC_STORE_IMPL(TYPE, TYPE_MANGLED, 0, TYPE_NV, TYPE_MANGLED_NV,\
                               , P, _gen_)

__CLC_NVVM_ATOMIC_STORE(int, i, int, i)
__CLC_NVVM_ATOMIC_STORE(uint, j, int, i)
__CLC_NVVM_ATOMIC_STORE(long, l, long, l)
__CLC_NVVM_ATOMIC_STORE(ulong, m, long, l)

__CLC_NVVM_ATOMIC_STORE(float, f, float, f)
#ifdef cl_khr_int64_base_atomics
__CLC_NVVM_ATOMIC_STORE(double, d, double, d)
#endif

#undef __CLC_NVVM_ATOMIC_STORE_TYPES
#undef __CLC_NVVM_ATOMIC_STORE
#undef __CLC_NVVM_ATOMIC_STORE_IMPL
