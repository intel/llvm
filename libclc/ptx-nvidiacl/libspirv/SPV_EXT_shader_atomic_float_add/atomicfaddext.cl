//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

// TODO: Convert scope to LLVM IR syncscope if __CUDA_ARCH >= sm_60
// TODO: Error if scope is not relaxed and __CUDA_ARCH <= sm_60
#define __CLC_ATOMICFADDEXT(TYPE, AS)                                          \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_AtomicFAddEXT(                           \
      __##AS TYPE *pointer, unsigned int scope, unsigned int semantics,        \
      TYPE value) {                                                            \
    /* Semantics mask may include memory order, storage class and other info   \
       Memory order is stored in the lowest 5 bits */                          \
    unsigned int order = semantics & 0x1F;                                     \
                                                                               \
    switch (order) {                                                           \
    case None:                                                                 \
      return __clc__atomic_fetch_add_##TYPE##_##AS##_relaxed(pointer, value);  \
    case Acquire:                                                              \
      return __clc__atomic_fetch_add_##TYPE##_##AS##_acquire(pointer, value);  \
    case Release:                                                              \
      return __clc__atomic_fetch_add_##TYPE##_##AS##_release(pointer, value);  \
    case AcquireRelease:                                                       \
      return __clc__atomic_fetch_add_##TYPE##_##AS##_acq_rel(pointer, value);  \
    default:                                                                   \
      /* Sequentially consistent atomics should never be incorrect */          \
    case SequentiallyConsistent:                                               \
      return __clc__atomic_fetch_add_##TYPE##_##AS##_seq_cst(pointer, value);  \
    }                                                                          \
  }

// FP32 atomics - must work without additional extensions
float __clc__atomic_fetch_add_float_global_relaxed(
    __global float *,
    float) __asm("__clc__atomic_fetch_add_float_global_relaxed");
float __clc__atomic_fetch_add_float_global_acquire(
    __global float *,
    float) __asm("__clc__atomic_fetch_add_float_global_acquire");
float __clc__atomic_fetch_add_float_global_release(
    __global float *,
    float) __asm("__clc__atomic_fetch_add_float_global_release");
float __clc__atomic_fetch_add_float_global_acq_rel(
    __global float *,
    float) __asm("__clc__atomic_fetch_add_float_global_acq_rel");
float __clc__atomic_fetch_add_float_global_seq_cst(
    __global float *,
    float) __asm("__clc__atomic_fetch_add_float_global_seq_cst");
float __clc__atomic_fetch_add_float_local_relaxed(__local float *, float) __asm(
    "__clc__atomic_fetch_add_float_local_relaxed");
float __clc__atomic_fetch_add_float_local_acquire(__local float *, float) __asm(
    "__clc__atomic_fetch_add_float_local_acquire");
float __clc__atomic_fetch_add_float_local_release(__local float *, float) __asm(
    "__clc__atomic_fetch_add_float_local_release");
float __clc__atomic_fetch_add_float_local_acq_rel(__local float *, float) __asm(
    "__clc__atomic_fetch_add_float_local_acq_rel");
float __clc__atomic_fetch_add_float_local_seq_cst(__local float *, float) __asm(
    "__clc__atomic_fetch_add_float_local_seq_cst");

__CLC_ATOMICFADDEXT(float, global)
__CLC_ATOMICFADDEXT(float, local)

_CLC_DECL float
_Z21__spirv_AtomicFAddEXTPU3AS1fN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEf(
    __global float *pointer, unsigned int scope, unsigned int semantics,
    float value) {
  return __spirv_AtomicFAddEXT(pointer, scope, semantics, value);
}

_CLC_DECL float
_Z21__spirv_AtomicFAddEXTPU3AS3fN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEf(
    __local float *pointer, unsigned int scope, unsigned int semantics,
    float value) {
  return __spirv_AtomicFAddEXT(pointer, scope, semantics, value);
}

// FP64 atomics - require cl_khr_fp64 extension
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
double __clc__atomic_fetch_add_double_global_relaxed(
    __global double *,
    double) __asm("__clc__atomic_fetch_add_double_global_relaxed");
double __clc__atomic_fetch_add_double_global_acquire(
    __global double *,
    double) __asm("__clc__atomic_fetch_add_double_global_acquire");
double __clc__atomic_fetch_add_double_global_release(
    __global double *,
    double) __asm("__clc__atomic_fetch_add_double_global_release");
double __clc__atomic_fetch_add_double_global_acq_rel(
    __global double *,
    double) __asm("__clc__atomic_fetch_add_double_global_acq_rel");
double __clc__atomic_fetch_add_double_global_seq_cst(
    __global double *,
    double) __asm("__clc__atomic_fetch_add_double_global_seq_cst");
double __clc__atomic_fetch_add_double_local_relaxed(
    __local double *,
    double) __asm("__clc__atomic_fetch_add_double_local_relaxed");
double __clc__atomic_fetch_add_double_local_acquire(
    __local double *,
    double) __asm("__clc__atomic_fetch_add_double_local_acquire");
double __clc__atomic_fetch_add_double_local_release(
    __local double *,
    double) __asm("__clc__atomic_fetch_add_double_local_release");
double __clc__atomic_fetch_add_double_local_acq_rel(
    __local double *,
    double) __asm("__clc__atomic_fetch_add_double_local_acq_rel");
double __clc__atomic_fetch_add_double_local_seq_cst(
    __local double *,
    double) __asm("__clc__atomic_fetch_add_double_local_seq_cst");

__CLC_ATOMICFADDEXT(double, global)
__CLC_ATOMICFADDEXT(double, local)

_CLC_DECL double
_Z21__spirv_AtomicFAddEXTPU3AS1dN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEd(
    __global double *pointer, unsigned int scope, unsigned int semantics,
    double value) {
  // FIXME: Double-precision atomics must be emulated for __CUDA_ARCH <= sm_50
  return __spirv_AtomicFAddEXT(pointer, scope, semantics, value);
}

_CLC_DECL double
_Z21__spirv_AtomicFAddEXTPU3AS3dN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEd(
    __local double *pointer, unsigned int scope, unsigned int semantics,
    double value) {
  // FIXME: Double-precision atomics must be emulated for __CUDA_ARCH <= sm_50
  return __spirv_AtomicFAddEXT(pointer, scope, semantics, value);
}
#endif // cl_khr_fp64

#undef __CLC_ATOMICFADDEXT
