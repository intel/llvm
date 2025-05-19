//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "atomic_helpers.h"
#include <libspirv/spirv.h>
#include <libspirv/spirv_types.h>

extern constant int __oclc_ISA_version;

AMDGPU_ATOMIC(__spirv_AtomicIAdd, int, __hip_atomic_fetch_add)
AMDGPU_ATOMIC(__spirv_AtomicIAdd, long, __hip_atomic_fetch_add)

#define AMDGPU_ATOMIC_FP32_ADD_IMPL(AS, CHECK, NEW_BUILTIN)                    \
  _CLC_OVERLOAD _CLC_DECL float __spirv_AtomicFAddEXT(                         \
      AS float *p, int scope, int semantics, float val) {                      \
    if (CHECK)                                                                 \
      return NEW_BUILTIN(p, val);                                              \
    int atomic_scope = 0, memory_order = 0;                                    \
    GET_ATOMIC_SCOPE_AND_ORDER(scope, atomic_scope, semantics, memory_order)   \
    return __hip_atomic_fetch_add(p, val, memory_order, atomic_scope);         \
  }

// Global AS atomics can be unsafe for malloc shared atomics, so should be opt
// in
AMDGPU_ATOMIC_FP32_ADD_IMPL(
    global,
    AMDGPU_ARCH_BETWEEN(9010, 10000) &&
        __oclc_amdgpu_reflect("AMDGPU_OCLC_UNSAFE_FP_ATOMICS"),
    __builtin_amdgcn_global_atomic_fadd_f32)
AMDGPU_ATOMIC_FP32_ADD_IMPL(local, AMDGPU_ARCH_GEQ(8000),
                            __builtin_amdgcn_ds_atomic_fadd_f32)
AMDGPU_ATOMIC_FP32_ADD_IMPL(, AMDGPU_ARCH_BETWEEN(9400, 10000),
                            __builtin_amdgcn_flat_atomic_fadd_f32)

#define AMDGPU_ATOMIC_FP64_ADD_IMPL(AS, CHECK, NEW_BUILTIN)                    \
  _CLC_OVERLOAD _CLC_DECL double __spirv_AtomicFAddEXT(                        \
      AS double *p, int scope, int semantics, double val) {                    \
    if (CHECK)                                                                 \
      return NEW_BUILTIN(p, val);                                              \
    int atomic_scope = 0, memory_order = 0;                                    \
    AS long *int_pointer = (AS long *)p;                                       \
    long old_int_val = 0, new_int_val = 0;                                     \
    do {                                                                       \
      old_int_val = __spirv_AtomicLoad(int_pointer, scope, semantics);         \
      double new_double_val = *(double *)&old_int_val + val;                   \
      new_int_val = *(long *)&new_double_val;                                  \
    } while (__spirv_AtomicCompareExchange(int_pointer, scope, semantics,      \
                                           semantics, new_int_val,             \
                                           old_int_val) != old_int_val);       \
                                                                               \
    return *(double *)&old_int_val;                                            \
  }

#ifdef cl_khr_int64_base_atomics
// Global AS atomics can be unsafe for malloc shared atomics, so should be opt
// in
AMDGPU_ATOMIC_FP64_ADD_IMPL(
    global,
    AMDGPU_ARCH_BETWEEN(9010, 10000) &&
        __oclc_amdgpu_reflect("AMDGPU_OCLC_UNSAFE_FP_ATOMICS"),
    __builtin_amdgcn_global_atomic_fadd_f64)
AMDGPU_ATOMIC_FP64_ADD_IMPL(local, AMDGPU_ARCH_BETWEEN(9010, 10000),
                            __builtin_amdgcn_ds_atomic_fadd_f64)
AMDGPU_ATOMIC_FP64_ADD_IMPL(, AMDGPU_ARCH_BETWEEN(9400, 10000),
                            __builtin_amdgcn_flat_atomic_fadd_f64)
#endif

#undef AMDGPU_ATOMIC
#undef AMDGPU_ATOMIC_IMPL
#undef AMDGPU_ATOMIC_FP32_ADD_IMPL
#undef AMDGPU_ATOMIC_FP64_ADD_IMPL
#undef AMDGPU_ARCH_GEQ
#undef AMDGPU_ARCH_BETWEEN
#undef GET_ATOMIC_SCOPE_AND_ORDER
