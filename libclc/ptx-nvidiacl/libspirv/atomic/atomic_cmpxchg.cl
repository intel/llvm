//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

int __clc_nvvm_reflect_arch();

#define __CLC_NVVM_ATOMIC_CAS_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP,   \
                                         ADDR_SPACE, ADDR_SPACE_NV, ORDER)     \
  switch (scope) {                                                             \
  case Subgroup:                                                               \
  case Workgroup: {                                                            \
    if (__clc_nvvm_reflect_arch() >= 600) {                                    \
      TYPE_NV res =                                                            \
          __nvvm_atom##ORDER##_cta_##OP##ADDR_SPACE_NV##TYPE_MANGLED_NV(       \
              (ADDR_SPACE TYPE_NV *)pointer, *(TYPE_NV *)&value, cmp);         \
      return *(TYPE *)&res;                                                    \
    }                                                                          \
  }                                                                            \
  case Device: {                                                               \
    TYPE_NV res = __nvvm_atom##ORDER##_##OP##ADDR_SPACE_NV##TYPE_MANGLED_NV(   \
        (ADDR_SPACE TYPE_NV *)pointer, *(TYPE_NV *)&value, cmp);               \
    return *(TYPE *)&res;                                                      \
  }                                                                            \
  case CrossDevice:                                                            \
  default: {                                                                   \
    if (__clc_nvvm_reflect_arch() >= 600) {                                    \
      TYPE_NV res =                                                            \
          __nvvm_atom##ORDER##_sys_##OP##ADDR_SPACE_NV##TYPE_MANGLED_NV(       \
              (ADDR_SPACE TYPE_NV *)pointer, *(TYPE_NV *)&value, cmp);         \
      return *(TYPE *)&res;                                                    \
    }                                                                          \
  }                                                                            \
  }

#define __CLC_NVVM_ATOMIC_CAS_IMPL(FN_MANGLED, TYPE, TYPE_MANGLED, TYPE_NV,    \
                                   TYPE_MANGLED_NV, OP, ADDR_SPACE,            \
                                   ADDR_SPACE_NV)                              \
  __attribute__((always_inline)) _CLC_DECL TYPE FN_MANGLED(                    \
      volatile ADDR_SPACE TYPE *pointer, enum Scope scope,                     \
      enum MemorySemanticsMask semantics1,                                     \
      enum MemorySemanticsMask semantics2, TYPE cmp, TYPE value) {             \
    /* Semantics mask may include memory order, storage class and other info   \
Memory order is stored in the lowest 5 bits */                                 \
    unsigned int order = (semantics1 | semantics2) & 0x1F;                     \
    switch (order) {                                                           \
    case None:                                                                 \
      __CLC_NVVM_ATOMIC_CAS_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP,     \
                                       ADDR_SPACE, ADDR_SPACE_NV, )            \
    case Acquire:                                                              \
      if (__clc_nvvm_reflect_arch() >= 700) {                                  \
        __CLC_NVVM_ATOMIC_CAS_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP,   \
                                         ADDR_SPACE, ADDR_SPACE_NV, _acquire)  \
      }                                                                        \
    case Release:                                                              \
      if (__clc_nvvm_reflect_arch() >= 700) {                                  \
        __CLC_NVVM_ATOMIC_CAS_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP,   \
                                         ADDR_SPACE, ADDR_SPACE_NV, _release)  \
      }                                                                        \
    case AcquireRelease:                                                       \
      if (__clc_nvvm_reflect_arch() >= 700) {                                  \
        __CLC_NVVM_ATOMIC_CAS_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP,   \
                                         ADDR_SPACE, ADDR_SPACE_NV, _acq_rel)  \
      }                                                                        \
    }                                                                          \
    __builtin_trap();                                                          \
    __builtin_unreachable();                                                   \
  }

#define __CLC_NVVM_ATOMIC_CAS(TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV,                                                                       \
                              OP)                                                                                                     \
  __CLC_NVVM_ATOMIC_CAS_IMPL(                                                                                                                     \
      _Z29__spirv_AtomicCompareExchange##P##TYPE_MANGLED##N5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagES4_##TYPE_MANGLED##TYPE_MANGLED,      \
      TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV, OP, , _gen_)                                                                                  \
  __CLC_NVVM_ATOMIC_CAS_IMPL(                                                                                                                     \
      _Z29__spirv_AtomicCompareExchange##PU3AS1##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_##TYPE_MANGLED##TYPE_MANGLED, \
      TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV, OP, __global, _global_)                                                                       \
  __CLC_NVVM_ATOMIC_CAS_IMPL(                                                                                                                     \
      _Z29__spirv_AtomicCompareExchange##PU3AS3##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_##TYPE_MANGLED##TYPE_MANGLED, \
      TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV, OP, __local, _shared_)

__CLC_NVVM_ATOMIC_CAS(int, i, int, i, cas)
__CLC_NVVM_ATOMIC_CAS(long, l, long, l, cas)
__CLC_NVVM_ATOMIC_CAS(unsigned int, j, int, i, cas)
__CLC_NVVM_ATOMIC_CAS(unsigned long, m, long, l, cas)

#undef __CLC_NVVM_ATOMIC_CAS_IMPL_ORDER
#undef __CLC_NVVM_ATOMIC_CAS
#undef __CLC_NVVM_ATOMIC_CAS_IMPL
