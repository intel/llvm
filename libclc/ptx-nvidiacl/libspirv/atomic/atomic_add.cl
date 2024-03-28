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

__CLC_NVVM_ATOMIC(int, i, int, i, add, _Z18__spirv_AtomicIAdd)
__CLC_NVVM_ATOMIC(uint, j, int, i, add, _Z18__spirv_AtomicIAdd)
__CLC_NVVM_ATOMIC(long, l, long, l, add, _Z18__spirv_AtomicIAdd)
__CLC_NVVM_ATOMIC(ulong, m, long, l, add, _Z18__spirv_AtomicIAdd)

__CLC_NVVM_ATOMIC(float, f, float, f, add, _Z21__spirv_AtomicFAddEXT)
#ifdef cl_khr_int64_base_atomics

#define __CLC_NVVM_ATOMIC_ADD_DOUBLE_IMPL(ADDR_SPACE, ADDR_SPACE_MANGLED,                                                                                     \
                                          ADDR_SPACE_NV, SUBSTITUTION1,                                                                                       \
                                          SUBSTITUTION2, SUBSTITUTION3)                                                                                       \
  long                                                                                                                                                        \
      _Z18__spirv_AtomicLoadP##ADDR_SPACE_MANGLED##KlN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(                                                      \
          volatile ADDR_SPACE const long *, enum Scope,                                                                                                       \
          enum MemorySemanticsMask);                                                                                                                          \
  long                                                                                                                                                        \
      _Z29__spirv_AtomicCompareExchange##P##ADDR_SPACE_MANGLED##lN5__spv5Scope4FlagENS##SUBSTITUTION1##_19MemorySemanticsMask4FlagES##SUBSTITUTION2##_ll(     \
          volatile ADDR_SPACE long *, enum Scope, enum MemorySemanticsMask,                                                                                   \
          enum MemorySemanticsMask, long, long);                                                                                                              \
  __attribute__((always_inline)) _CLC_DECL double                                                                                                             \
      _Z21__spirv_AtomicFAddEXT##P##ADDR_SPACE_MANGLED##d##N5__spv5Scope4FlagENS##SUBSTITUTION3##_19MemorySemanticsMask4FlagE##d(                             \
          volatile ADDR_SPACE double *pointer, enum Scope scope,                                                                                              \
          enum MemorySemanticsMask semantics, double value) {                                                                                                 \
    /* Semantics mask may include memory order, storage class and other info                                                                                  \
Memory order is stored in the lowest 5 bits */                                                                                                                \
    unsigned int order = semantics & 0x1F;                                                                                                                    \
    if (__clc_nvvm_reflect_arch() >= 600) {                                                                                                                   \
      switch (order) {                                                                                                                                        \
      case None:                                                                                                                                              \
        __CLC_NVVM_ATOMIC_IMPL_ORDER(double, double, d, add, ADDR_SPACE,                                                                                      \
                                     ADDR_SPACE_NV, )                                                                                                         \
        break;                                                                                                                                                \
      case Acquire:                                                                                                                                           \
        if (__clc_nvvm_reflect_arch() >= 700) {                                                                                                               \
          __CLC_NVVM_ATOMIC_IMPL_ORDER(double, double, d, add, ADDR_SPACE,                                                                                    \
                                       ADDR_SPACE_NV, _acquire)                                                                                               \
        } else {                                                                                                                                              \
          __CLC_NVVM_ATOMIC_IMPL_ACQUIRE_FENCE(double, double, d, add,                                                                                        \
                                               ADDR_SPACE, ADDR_SPACE_NV)                                                                                     \
        }                                                                                                                                                     \
        break;                                                                                                                                                \
      case Release:                                                                                                                                           \
        if (__clc_nvvm_reflect_arch() >= 700) {                                                                                                               \
          __CLC_NVVM_ATOMIC_IMPL_ORDER(double, double, d, add, ADDR_SPACE,                                                                                    \
                                       ADDR_SPACE_NV, _release)                                                                                               \
        } else {                                                                                                                                              \
          __spirv_MemoryBarrier(scope, Release);                                                                                                              \
          __CLC_NVVM_ATOMIC_IMPL_ORDER(double, double, d, add, ADDR_SPACE,                                                                                    \
                                       ADDR_SPACE_NV, )                                                                                                       \
        }                                                                                                                                                     \
        break;                                                                                                                                                \
      case AcquireRelease:                                                                                                                                    \
        if (__clc_nvvm_reflect_arch() >= 700) {                                                                                                               \
          __CLC_NVVM_ATOMIC_IMPL_ORDER(double, double, d, add, ADDR_SPACE,                                                                                    \
                                       ADDR_SPACE_NV, _acq_rel)                                                                                               \
        } else {                                                                                                                                              \
          __spirv_MemoryBarrier(scope, Release);                                                                                                              \
          __CLC_NVVM_ATOMIC_IMPL_ACQUIRE_FENCE(double, double, d, add,                                                                                        \
                                               ADDR_SPACE, ADDR_SPACE_NV)                                                                                     \
        }                                                                                                                                                     \
        break;                                                                                                                                                \
      case SequentiallyConsistent:                                                                                                                            \
        if (__clc_nvvm_reflect_arch() >= 700) {                                                                                                               \
          __CLC_NVVM_FENCE_SC_SM70()                                                                                                                          \
          __CLC_NVVM_ATOMIC_IMPL_ORDER(double, double, d, add, ADDR_SPACE,                                                                                    \
                                       ADDR_SPACE_NV, _acq_rel)                                                                                               \
          break;                                                                                                                                              \
        }                                                                                                                                                     \
      }                                                                                                                                                       \
      __builtin_trap();                                                                                                                                       \
      __builtin_unreachable();                                                                                                                                \
    } else {                                                                                                                                                  \
      enum MemorySemanticsMask load_order;                                                                                                                    \
      switch (semantics) {                                                                                                                                    \
      case SequentiallyConsistent:                                                                                                                            \
        load_order = SequentiallyConsistent;                                                                                                                  \
        break;                                                                                                                                                \
      case Acquire:                                                                                                                                           \
      case AcquireRelease:                                                                                                                                    \
        load_order = Acquire;                                                                                                                                 \
        break;                                                                                                                                                \
      default:                                                                                                                                                \
        load_order = None;                                                                                                                                    \
      }                                                                                                                                                       \
      volatile ADDR_SPACE long *pointer_int =                                                                                                                 \
          (volatile ADDR_SPACE long *)pointer;                                                                                                                \
      long old_int;                                                                                                                                           \
      long new_val_int;                                                                                                                                       \
      do {                                                                                                                                                    \
        old_int =                                                                                                                                             \
            _Z18__spirv_AtomicLoadP##ADDR_SPACE_MANGLED##KlN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(                                                \
                pointer_int, scope, load_order);                                                                                                              \
        double new_val = *(double *)&old_int + *(double *)&value;                                                                                             \
        new_val_int = *(long *)&new_val;                                                                                                                      \
      } while (                                                                                                                                               \
          _Z29__spirv_AtomicCompareExchange##P##ADDR_SPACE_MANGLED##lN5__spv5Scope4FlagENS##SUBSTITUTION1##_19MemorySemanticsMask4FlagES##SUBSTITUTION2##_ll( \
              pointer_int, scope, semantics, semantics, new_val_int,                                                                                          \
              old_int) != old_int);                                                                                                                           \
      return *(double *)&old_int;                                                                                                                             \
    }                                                                                                                                                         \
  }

__CLC_NVVM_ATOMIC_ADD_DOUBLE_IMPL(, , _gen_, 0, 4, 0)
__CLC_NVVM_ATOMIC_ADD_DOUBLE_IMPL(__global, U3AS1, _global_, 1, 5, 1)
__CLC_NVVM_ATOMIC_ADD_DOUBLE_IMPL(__local, U3AS3, _shared_, 1, 5, 1)

#endif

#undef __CLC_NVVM_ATOMIC_TYPES
#undef __CLC_NVVM_ATOMIC
#undef __CLC_NVVM_ATOMIC_IMPL
