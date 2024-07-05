//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "atomic_helpers.h"
#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

extern constant int __oclc_ISA_version;

AMDGPU_ATOMIC(_Z18__spirv_AtomicIAdd, int, i, __hip_atomic_fetch_add)
AMDGPU_ATOMIC(_Z18__spirv_AtomicIAdd, unsigned int, j, __hip_atomic_fetch_add)
AMDGPU_ATOMIC(_Z18__spirv_AtomicIAdd, long, l, __hip_atomic_fetch_add)
AMDGPU_ATOMIC(_Z18__spirv_AtomicIAdd, unsigned long, m, __hip_atomic_fetch_add)

#define AMDGPU_ATOMIC_FP32_ADD_IMPL(AS, AS_MANGLED, SUB1, CHECK, NEW_BUILTIN)                              \
  _CLC_DEF float                                                                                           \
      _Z21__spirv_AtomicFAddEXTP##AS_MANGLED##fN5__spv5Scope4FlagENS##SUB1##_19MemorySemanticsMask4FlagEf( \
          volatile AS float *p, enum Scope scope,                                                          \
          enum MemorySemanticsMask semantics, float val) {                                                 \
    if (CHECK)                                                                                             \
      return NEW_BUILTIN(p, val);                                                                          \
    int atomic_scope = 0, memory_order = 0;                                                                \
    GET_ATOMIC_SCOPE_AND_ORDER(scope, atomic_scope, semantics, memory_order)                               \
    return __hip_atomic_fetch_add(p, val, memory_order, atomic_scope);                                     \
  }

// Global AS atomics can be unsafe for malloc shared atomics, so should be opt
// in
AMDGPU_ATOMIC_FP32_ADD_IMPL(
    global, U3AS1, 1,
    AMDGPU_ARCH_BETWEEN(9010, 10000) &&
        __oclc_amdgpu_reflect("AMDGPU_OCLC_UNSAFE_FP_ATOMICS"),
    __builtin_amdgcn_global_atomic_fadd_f32)
AMDGPU_ATOMIC_FP32_ADD_IMPL(local, U3AS3, 1, AMDGPU_ARCH_GEQ(8000),
                            __builtin_amdgcn_ds_atomic_fadd_f32)
AMDGPU_ATOMIC_FP32_ADD_IMPL(, , 0, AMDGPU_ARCH_BETWEEN(9400, 10000),
                            __builtin_amdgcn_flat_atomic_fadd_f32)

#define AMDGPU_ATOMIC_FP64_ADD_IMPL(AS, AS_MANGLED, SUB1, SUB2, CHECK,                                                          \
                                    NEW_BUILTIN)                                                                                \
  _CLC_DEF long                                                                                                                 \
      _Z29__spirv_AtomicCompareExchangeP##AS_MANGLED##lN5__spv5Scope4FlagENS##SUB1##_19MemorySemanticsMask4FlagES##SUB2##_ll(   \
          volatile AS long *, enum Scope, enum MemorySemanticsMask,                                                             \
          enum MemorySemanticsMask, long desired, long expected);                                                               \
  _CLC_DEF long                                                                                                                 \
      _Z18__spirv_AtomicLoadP##AS_MANGLED##KlN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(                                \
          const volatile AS long *, enum Scope, enum MemorySemanticsMask);                                                      \
  _CLC_DEF double                                                                                                               \
      _Z21__spirv_AtomicFAddEXTP##AS_MANGLED##dN5__spv5Scope4FlagENS##SUB1##_19MemorySemanticsMask4FlagEd(                      \
          volatile AS double *p, enum Scope scope,                                                                              \
          enum MemorySemanticsMask semantics, double val) {                                                                     \
    if (CHECK)                                                                                                                  \
      return NEW_BUILTIN(p, val);                                                                                               \
    int atomic_scope = 0, memory_order = 0;                                                                                     \
    volatile AS long *int_pointer = (volatile AS long *)p;                                                                      \
    long old_int_val = 0, new_int_val = 0;                                                                                      \
    do {                                                                                                                        \
      old_int_val =                                                                                                             \
          _Z18__spirv_AtomicLoadP##AS_MANGLED##KlN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(                            \
              int_pointer, scope, semantics);                                                                                   \
      double new_double_val = *(double *)&old_int_val + val;                                                                    \
      new_int_val = *(long *)&new_double_val;                                                                                   \
    } while (                                                                                                                   \
        _Z29__spirv_AtomicCompareExchangeP##AS_MANGLED##lN5__spv5Scope4FlagENS##SUB1##_19MemorySemanticsMask4FlagES##SUB2##_ll( \
            int_pointer, scope, semantics, semantics, new_int_val,                                                              \
            old_int_val) != old_int_val);                                                                                       \
                                                                                                                                \
    return *(double *)&old_int_val;                                                                                             \
  }

#ifdef cl_khr_int64_base_atomics
// Global AS atomics can be unsafe for malloc shared atomics, so should be opt
// in
AMDGPU_ATOMIC_FP64_ADD_IMPL(
    global, U3AS1, 1, 5,
    AMDGPU_ARCH_BETWEEN(9010, 10000) &&
        __oclc_amdgpu_reflect("AMDGPU_OCLC_UNSAFE_FP_ATOMICS"),
    __builtin_amdgcn_global_atomic_fadd_f64)
AMDGPU_ATOMIC_FP64_ADD_IMPL(local, U3AS3, 1, 5,
                            AMDGPU_ARCH_BETWEEN(9010, 10000),
                            __builtin_amdgcn_ds_atomic_fadd_f64)
AMDGPU_ATOMIC_FP64_ADD_IMPL(, , 0, 4, AMDGPU_ARCH_BETWEEN(9400, 10000),
                            __builtin_amdgcn_flat_atomic_fadd_f64)
#endif

#undef AMDGPU_ATOMIC
#undef AMDGPU_ATOMIC_IMPL
#undef AMDGPU_ATOMIC_FP32_ADD_IMPL
#undef AMDGPU_ATOMIC_FP64_ADD_IMPL
#undef AMDGPU_ARCH_GEQ
#undef AMDGPU_ARCH_BETWEEN
#undef GET_ATOMIC_SCOPE_AND_ORDER
