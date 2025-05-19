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

__CLC_NVVM_ATOMIC(int, int, i, add, __spirv_AtomicIAdd)
__CLC_NVVM_ATOMIC(long, long, l, add, __spirv_AtomicIAdd)

__CLC_NVVM_ATOMIC(float, float, f, add, __spirv_AtomicFAddEXT)
#ifdef cl_khr_int64_base_atomics

#define __CLC_NVVM_ATOMIC_ADD_DOUBLE_IMPL(ADDR_SPACE, ADDR_SPACE_NV)           \
  __attribute__((always_inline)) _CLC_OVERLOAD _CLC_DECL double                \
  __spirv_AtomicFAddEXT(ADDR_SPACE double *pointer, int scope, int semantics,  \
                        double value) {                                        \
    /* Semantics mask may include memory order, storage class and other info   \
Memory order is stored in the lowest 5 bits */                                 \
    unsigned int order = semantics & 0x1F;                                     \
    if (__clc_nvvm_reflect_arch() >= 600) {                                    \
      switch (order) {                                                         \
      case None:                                                               \
        __CLC_NVVM_ATOMIC_IMPL_ORDER(double, double, d, add, ADDR_SPACE,       \
                                     ADDR_SPACE_NV, )                          \
        break;                                                                 \
      case Acquire:                                                            \
        if (__clc_nvvm_reflect_arch() >= 700) {                                \
          __CLC_NVVM_ATOMIC_IMPL_ORDER(double, double, d, add, ADDR_SPACE,     \
                                       ADDR_SPACE_NV, _acquire)                \
        } else {                                                               \
          __CLC_NVVM_ATOMIC_IMPL_ACQUIRE_FENCE(double, double, d, add,         \
                                               ADDR_SPACE, ADDR_SPACE_NV)      \
        }                                                                      \
        break;                                                                 \
      case Release:                                                            \
        if (__clc_nvvm_reflect_arch() >= 700) {                                \
          __CLC_NVVM_ATOMIC_IMPL_ORDER(double, double, d, add, ADDR_SPACE,     \
                                       ADDR_SPACE_NV, _release)                \
        } else {                                                               \
          __spirv_MemoryBarrier(scope, Release);                               \
          __CLC_NVVM_ATOMIC_IMPL_ORDER(double, double, d, add, ADDR_SPACE,     \
                                       ADDR_SPACE_NV, )                        \
        }                                                                      \
        break;                                                                 \
      case AcquireRelease:                                                     \
        if (__clc_nvvm_reflect_arch() >= 700) {                                \
          __CLC_NVVM_ATOMIC_IMPL_ORDER(double, double, d, add, ADDR_SPACE,     \
                                       ADDR_SPACE_NV, _acq_rel)                \
        } else {                                                               \
          __spirv_MemoryBarrier(scope, Release);                               \
          __CLC_NVVM_ATOMIC_IMPL_ACQUIRE_FENCE(double, double, d, add,         \
                                               ADDR_SPACE, ADDR_SPACE_NV)      \
        }                                                                      \
        break;                                                                 \
      case SequentiallyConsistent:                                             \
        if (__clc_nvvm_reflect_arch() >= 700) {                                \
          __CLC_NVVM_FENCE_SC_SM70()                                           \
          __CLC_NVVM_ATOMIC_IMPL_ORDER(double, double, d, add, ADDR_SPACE,     \
                                       ADDR_SPACE_NV, _acq_rel)                \
          break;                                                               \
        }                                                                      \
      }                                                                        \
      __builtin_trap();                                                        \
      __builtin_unreachable();                                                 \
    } else {                                                                   \
      int load_order;                                                          \
      switch (semantics) {                                                     \
      case SequentiallyConsistent:                                             \
        load_order = SequentiallyConsistent;                                   \
        break;                                                                 \
      case Acquire:                                                            \
      case AcquireRelease:                                                     \
        load_order = Acquire;                                                  \
        break;                                                                 \
      default:                                                                 \
        load_order = None;                                                     \
      }                                                                        \
      ADDR_SPACE long *pointer_int = (ADDR_SPACE long *)pointer;               \
      long old_int;                                                            \
      long new_val_int;                                                        \
      do {                                                                     \
        old_int = __spirv_AtomicLoad(pointer_int, scope, load_order);          \
        double new_val = *(double *)&old_int + *(double *)&value;              \
        new_val_int = *(long *)&new_val;                                       \
      } while (__spirv_AtomicCompareExchange(pointer_int, scope, semantics,    \
                                             semantics, new_val_int,           \
                                             old_int) != old_int);             \
      return *(double *)&old_int;                                              \
    }                                                                          \
  }

__CLC_NVVM_ATOMIC_ADD_DOUBLE_IMPL(, _gen_)
__CLC_NVVM_ATOMIC_ADD_DOUBLE_IMPL(__global, _global_)
__CLC_NVVM_ATOMIC_ADD_DOUBLE_IMPL(__local, _shared_)

#endif

#undef __CLC_NVVM_ATOMIC_TYPES
#undef __CLC_NVVM_ATOMIC
#undef __CLC_NVVM_ATOMIC_IMPL
