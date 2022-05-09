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

#define __CLC_NVVM_ATOMIC_IMPL(FN_MANGLED, TYPE, TYPE_MANGLED, TYPE_NV,        \
                               TYPE_MANGLED_NV, OP, ADDR_SPACE, ADDR_SPACE_NV) \
  __attribute__((always_inline)) _CLC_DECL TYPE FN_MANGLED(                    \
      volatile ADDR_SPACE TYPE *pointer, enum Scope scope,                     \
      enum MemorySemanticsMask semantics, TYPE value) {                        \
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
    }                                                                          \
    __builtin_trap();                                                          \
    __builtin_unreachable();                                                   \
  }

#define __CLC_NVVM_ATOMIC(TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV, OP,                                 \
                          NAME_MANGLED)                                                                     \
  __CLC_NVVM_ATOMIC_IMPL(                                                                                   \
      NAME_MANGLED##P##TYPE_MANGLED##N5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagE##TYPE_MANGLED,      \
      TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV, OP, , _gen_)                                            \
  __CLC_NVVM_ATOMIC_IMPL(                                                                                   \
      NAME_MANGLED##PU3AS1##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE##TYPE_MANGLED, \
      TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV, OP, __global, _global_)                                 \
  __CLC_NVVM_ATOMIC_IMPL(                                                                                   \
      NAME_MANGLED##PU3AS3##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE##TYPE_MANGLED, \
      TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV, OP, __local, _shared_)

#endif
