//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_ATOMIC_HELPERS_H
#define __CLC_ATOMIC_HELPERS_H

#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

extern int __clc_nvvm_reflect_arch();
_CLC_OVERLOAD _CLC_DECL void __spirv_MemoryBarrier(unsigned int, unsigned int);

#define __CLC_NVVM_ATOMIC_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP,       \
                                     ADDR_SPACE, ADDR_SPACE_NV, ORDER)         \
  switch (scope) {                                                             \
  case Subgroup:                                                               \
  case Workgroup: {                                                            \
    if (__clc_nvvm_reflect_arch() >= 600) {                                    \
      TYPE_NV res =                                                            \
          __nvvm_atom##ORDER##_cta_##OP##ADDR_SPACE_NV##TYPE_MANGLED_NV(       \
              (ADDR_SPACE TYPE_NV *)pointer, *(TYPE_NV *)&value);              \
      return *(TYPE *)&res;                                                    \
    }                                                                          \
  }                                                                            \
  case Device: {                                                               \
    TYPE_NV res = __nvvm_atom##ORDER##_##OP##ADDR_SPACE_NV##TYPE_MANGLED_NV(   \
        (ADDR_SPACE TYPE_NV *)pointer, *(TYPE_NV *)&value);                    \
    return *(TYPE *)&res;                                                      \
  }                                                                            \
  case CrossDevice:                                                            \
  default: {                                                                   \
    if (__clc_nvvm_reflect_arch() >= 600) {                                    \
      TYPE_NV res =                                                            \
          __nvvm_atom##ORDER##_sys_##OP##ADDR_SPACE_NV##TYPE_MANGLED_NV(       \
              (ADDR_SPACE TYPE_NV *)pointer, *(TYPE_NV *)&value);              \
      return *(TYPE *)&res;                                                    \
    }                                                                          \
  }                                                                            \
  }

#define __CLC_NVVM_ATOMIC_IMPL_ACQUIRE_FENCE(TYPE, TYPE_NV, TYPE_MANGLED_NV,   \
                                             OP, ADDR_SPACE, ADDR_SPACE_NV)    \
  switch (scope) {                                                             \
  case Subgroup:                                                               \
  case Workgroup: {                                                            \
    if (__clc_nvvm_reflect_arch() >= 600) {                                    \
      TYPE_NV res = __nvvm_atom##_cta_##OP##ADDR_SPACE_NV##TYPE_MANGLED_NV(    \
          (ADDR_SPACE TYPE_NV *)pointer, *(TYPE_NV *)&value);                  \
      __spirv_MemoryBarrier(Workgroup, Acquire);                               \
      return *(TYPE *)&res;                                                    \
    }                                                                          \
  }                                                                            \
  case Device: {                                                               \
    TYPE_NV res = __nvvm_atom##_##OP##ADDR_SPACE_NV##TYPE_MANGLED_NV(          \
        (ADDR_SPACE TYPE_NV *)pointer, *(TYPE_NV *)&value);                    \
    __spirv_MemoryBarrier(Device, Acquire);                                    \
    return *(TYPE *)&res;                                                      \
  }                                                                            \
  case CrossDevice:                                                            \
  default: {                                                                   \
    if (__clc_nvvm_reflect_arch() >= 600) {                                    \
      TYPE_NV res = __nvvm_atom##_sys_##OP##ADDR_SPACE_NV##TYPE_MANGLED_NV(    \
          (ADDR_SPACE TYPE_NV *)pointer, *(TYPE_NV *)&value);                  \
      __spirv_MemoryBarrier(CrossDevice, Acquire);                             \
      return *(TYPE *)&res;                                                    \
    }                                                                          \
  }                                                                            \
  }

#define __CLC_NVVM_FENCE_SC_SM70()                                             \
  if (scope == CrossDevice) {                                                  \
    __asm__ __volatile__("fence.sc.sys;");                                     \
  } else if (scope == Device) {                                                \
    __asm__ __volatile__("fence.sc.gpu;");                                     \
  } else {                                                                     \
    __asm__ __volatile__("fence.sc.cta;");                                     \
  }

#define __CLC_NVVM_ATOMIC_IMPL(                                                \
    TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV, OP, NAME_MANGLED,            \
    ADDR_SPACE, POINTER_AND_ADDR_SPACE_MANGLED, ADDR_SPACE_NV, SUBSTITUTION)   \
  __attribute__((always_inline)) _CLC_DECL TYPE                                \
      NAME_MANGLED##POINTER_AND_ADDR_SPACE_MANGLED##TYPE_MANGLED##N5__spv\
5Scope4FlagENS##SUBSTITUTION##_19MemorySemanticsMask4FlagE##TYPE_MANGLED(      \
          volatile ADDR_SPACE TYPE *pointer, enum Scope scope,                 \
          enum MemorySemanticsMask semantics, TYPE value) {                    \
    /* Semantics mask may include memory order, storage class and other info   \
Memory order is stored in the lowest 5 bits */                                 \
    unsigned int order = semantics & 0x1F;                                     \
    switch (order) {                                                           \
    case None:                                                                 \
      __CLC_NVVM_ATOMIC_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP,         \
                                   ADDR_SPACE, ADDR_SPACE_NV, )                \
      break;                                                                   \
    case Acquire:                                                              \
      if (__clc_nvvm_reflect_arch() >= 700) {                                  \
        __CLC_NVVM_ATOMIC_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP,       \
                                     ADDR_SPACE, ADDR_SPACE_NV, _acquire)      \
      } else {                                                                 \
        __CLC_NVVM_ATOMIC_IMPL_ACQUIRE_FENCE(TYPE, TYPE_NV, TYPE_MANGLED_NV,   \
                                             OP, ADDR_SPACE, ADDR_SPACE_NV)    \
      }                                                                        \
      break;                                                                   \
    case Release:                                                              \
      if (__clc_nvvm_reflect_arch() >= 700) {                                  \
        __CLC_NVVM_ATOMIC_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP,       \
                                     ADDR_SPACE, ADDR_SPACE_NV, _release)      \
      } else {                                                                 \
        __spirv_MemoryBarrier(scope, Release);                                 \
        __CLC_NVVM_ATOMIC_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP,       \
                                     ADDR_SPACE, ADDR_SPACE_NV, )              \
      }                                                                        \
      break;                                                                   \
    case AcquireRelease:                                                       \
      if (__clc_nvvm_reflect_arch() >= 700) {                                  \
        __CLC_NVVM_ATOMIC_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP,       \
                                     ADDR_SPACE, ADDR_SPACE_NV, _acq_rel)      \
      } else {                                                                 \
        __spirv_MemoryBarrier(scope, Release);                                 \
        __CLC_NVVM_ATOMIC_IMPL_ACQUIRE_FENCE(TYPE, TYPE_NV, TYPE_MANGLED_NV,   \
                                             OP, ADDR_SPACE, ADDR_SPACE_NV)    \
      }                                                                        \
      break;                                                                   \
    case SequentiallyConsistent:                                               \
      if (__clc_nvvm_reflect_arch() >= 700) {                                  \
        __CLC_NVVM_FENCE_SC_SM70()                                             \
        __CLC_NVVM_ATOMIC_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP,       \
                                     ADDR_SPACE, ADDR_SPACE_NV, _acq_rel)      \
        break;                                                                 \
      }                                                                        \
    }                                                                          \
    __builtin_trap();                                                          \
    __builtin_unreachable();                                                   \
  }

#define __CLC_NVVM_ATOMIC(TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV, OP,    \
                          NAME_MANGLED)                                        \
  __CLC_NVVM_ATOMIC_IMPL(TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV, OP,     \
                         NAME_MANGLED, __global, PU3AS1, _global_, 1)          \
  __CLC_NVVM_ATOMIC_IMPL(TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV, OP,     \
                         NAME_MANGLED, __local, PU3AS3, _shared_, 1)           \
  __CLC_NVVM_ATOMIC_IMPL(TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV, OP,     \
                         NAME_MANGLED, , P, _gen_, 0)
#endif
