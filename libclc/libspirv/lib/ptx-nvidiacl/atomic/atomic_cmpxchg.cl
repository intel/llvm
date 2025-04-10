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

int __clc_nvvm_reflect_arch();
_CLC_OVERLOAD _CLC_DECL void __spirv_MemoryBarrier(int, int);

#define __CLC_NVVM_ATOMIC_CAS_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP,   \
                                         ADDR_SPACE, ADDR_SPACE_NV, ORDER)     \
  switch (scope) {                                                             \
  case Invocation:                                                             \
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

#define __CLC_NVVM_ATOMIC_CAS_IMPL_ACQUIRE_FENCE(                              \
    TYPE, TYPE_NV, TYPE_MANGLED_NV, OP, ADDR_SPACE, ADDR_SPACE_NV)             \
  switch (scope) {                                                             \
  case Invocation:                                                             \
  case Subgroup:                                                               \
  case Workgroup: {                                                            \
    if (__clc_nvvm_reflect_arch() >= 600) {                                    \
      TYPE_NV res = __nvvm_atom##_cta_##OP##ADDR_SPACE_NV##TYPE_MANGLED_NV(    \
          (ADDR_SPACE TYPE_NV *)pointer, *(TYPE_NV *)&value, cmp);             \
      __spirv_MemoryBarrier(Workgroup, Acquire);                               \
      return *(TYPE *)&res;                                                    \
    }                                                                          \
  }                                                                            \
  case Device: {                                                               \
    TYPE_NV res = __nvvm_atom##_##OP##ADDR_SPACE_NV##TYPE_MANGLED_NV(          \
        (ADDR_SPACE TYPE_NV *)pointer, *(TYPE_NV *)&value, cmp);               \
    __spirv_MemoryBarrier(Device, Acquire);                                    \
    return *(TYPE *)&res;                                                      \
  }                                                                            \
  case CrossDevice:                                                            \
  default: {                                                                   \
    if (__clc_nvvm_reflect_arch() >= 600) {                                    \
      TYPE_NV res = __nvvm_atom##_sys_##OP##ADDR_SPACE_NV##TYPE_MANGLED_NV(    \
          (ADDR_SPACE TYPE_NV *)pointer, *(TYPE_NV *)&value, cmp);             \
      __spirv_MemoryBarrier(CrossDevice, Acquire);                             \
      return *(TYPE *)&res;                                                    \
    }                                                                          \
  }                                                                            \
  }

// Type __spirv_AtomicCompareExchange(AS Type *P, int S, int E, int U,
//                                    Type V, Type C);
#define __CLC_NVVM_ATOMIC_CAS_IMPL(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP,         \
                                   OP_MANGLED, ADDR_SPACE, ADDR_SPACE_NV)      \
  __attribute__((always_inline)) _CLC_OVERLOAD _CLC_DECL TYPE                  \
      __spirv_Atomic##OP_MANGLED(ADDR_SPACE TYPE *pointer, int scope,          \
                                 int semantics1, int semantics2, TYPE cmp,     \
                                 TYPE value) {                                 \
    /* Semantics mask may include memory order, storage class and other info   \
Memory order is stored in the lowest 5 bits */                                 \
    unsigned int order = semantics1 & 0x1F;                                    \
    switch (order) {                                                           \
    case None:                                                                 \
      __CLC_NVVM_ATOMIC_CAS_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP,     \
                                       ADDR_SPACE, ADDR_SPACE_NV, )            \
    case Acquire:                                                              \
      if (__clc_nvvm_reflect_arch() >= 700) {                                  \
        __CLC_NVVM_ATOMIC_CAS_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP,   \
                                         ADDR_SPACE, ADDR_SPACE_NV, _acquire)  \
      } else {                                                                 \
        __CLC_NVVM_ATOMIC_CAS_IMPL_ACQUIRE_FENCE(                              \
            TYPE, TYPE_NV, TYPE_MANGLED_NV, OP, ADDR_SPACE, ADDR_SPACE_NV)     \
      }                                                                        \
      break;                                                                   \
    case Release:                                                              \
      if (__clc_nvvm_reflect_arch() >= 700) {                                  \
        __CLC_NVVM_ATOMIC_CAS_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP,   \
                                         ADDR_SPACE, ADDR_SPACE_NV, _release)  \
      } else {                                                                 \
        __spirv_MemoryBarrier(scope, Release);                                 \
        __CLC_NVVM_ATOMIC_CAS_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP,   \
                                         ADDR_SPACE, ADDR_SPACE_NV, )          \
      }                                                                        \
      break;                                                                   \
    case AcquireRelease:                                                       \
      if (__clc_nvvm_reflect_arch() >= 700) {                                  \
        __CLC_NVVM_ATOMIC_CAS_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP,   \
                                         ADDR_SPACE, ADDR_SPACE_NV, _acq_rel)  \
      } else {                                                                 \
        __spirv_MemoryBarrier(scope, Release);                                 \
        __CLC_NVVM_ATOMIC_CAS_IMPL_ACQUIRE_FENCE(                              \
            TYPE, TYPE_NV, TYPE_MANGLED_NV, OP, ADDR_SPACE, ADDR_SPACE_NV)     \
      }                                                                        \
      break;                                                                   \
    case SequentiallyConsistent:                                               \
      if (__clc_nvvm_reflect_arch() >= 700) {                                  \
        __CLC_NVVM_FENCE_SC_SM70()                                             \
        __CLC_NVVM_ATOMIC_CAS_IMPL_ORDER(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP,   \
                                         ADDR_SPACE, ADDR_SPACE_NV, _acq_rel)  \
        break;                                                                 \
      }                                                                        \
    }                                                                          \
    __builtin_trap();                                                          \
    __builtin_unreachable();                                                   \
  }

#define __CLC_NVVM_ATOMIC_CAS(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP, OP_MANGLED)  \
  __CLC_NVVM_ATOMIC_CAS_IMPL(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP, OP_MANGLED,   \
                             __global, _global_)                               \
  __CLC_NVVM_ATOMIC_CAS_IMPL(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP, OP_MANGLED,   \
                             __local, _shared_)                                \
  __CLC_NVVM_ATOMIC_CAS_IMPL(TYPE, TYPE_NV, TYPE_MANGLED_NV, OP, OP_MANGLED, , \
                             _gen_)

__CLC_NVVM_ATOMIC_CAS(int, int, i, cas, CompareExchange)
__CLC_NVVM_ATOMIC_CAS(long, long, l, cas, CompareExchange)
__CLC_NVVM_ATOMIC_CAS(unsigned int, int, i, cas, CompareExchange)
__CLC_NVVM_ATOMIC_CAS(unsigned long, long, l, cas, CompareExchange)
__CLC_NVVM_ATOMIC_CAS(float, float, f, cas, CompareExchange)
__CLC_NVVM_ATOMIC_CAS(double, double, d, cas, CompareExchange)

#undef __CLC_NVVM_ATOMIC_CAS_IMPL_ORDER
#undef __CLC_NVVM_ATOMIC_CAS
#undef __CLC_NVVM_ATOMIC_CAS_IMPL
