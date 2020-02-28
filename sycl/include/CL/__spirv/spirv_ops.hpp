//==---------- spirv_ops.hpp --- SPIRV operations -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/__spirv/spirv_types.hpp>
#include <CL/sycl/detail/defines.hpp>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#ifdef __SYCL_DEVICE_ONLY__

template <typename RetT, typename ImageT>
extern SYCL_EXTERNAL RetT __spirv_ImageQueryFormat(ImageT);

template <typename RetT, typename ImageT>
extern SYCL_EXTERNAL RetT __spirv_ImageQueryOrder(ImageT);

template <typename RetT, typename ImageT>
extern SYCL_EXTERNAL RetT __spirv_ImageQuerySize(ImageT);

template <typename ImageT, typename CoordT, typename ValT>
extern SYCL_EXTERNAL void __spirv_ImageWrite(ImageT, CoordT, ValT);

template <class RetT, typename ImageT, typename TempArgT>
extern SYCL_EXTERNAL RetT __spirv_ImageRead(ImageT, TempArgT);

template <typename ImageT, typename SampledType>
extern SYCL_EXTERNAL SampledType __spirv_SampledImage(ImageT, __ocl_sampler_t);

template <typename SampledType, typename TempRetT, typename TempArgT>
extern SYCL_EXTERNAL TempRetT __spirv_ImageSampleExplicitLod(SampledType,
                                                             TempArgT, int,
                                                             float);

#ifdef __SYCL_NVPTX__

//
// This a workaround to avoid a SPIR-V ABI issue.
//

template <typename dataT>
__ocl_event_t __spirv_GroupAsyncCopy(__spv::Scope Execution,
                                     __attribute__((opencl_local)) dataT *Dest,
                                     __attribute__((opencl_global)) dataT *Src,
                                     size_t NumElements, size_t Stride,
                                     __ocl_event_t E) noexcept {
  for (int i = 0; i < NumElements; i++) {
    Dest[i] = Src[i * Stride];
  }

  return E;
}

template <typename dataT>
__ocl_event_t __spirv_GroupAsyncCopy(__spv::Scope Execution,
                                     __attribute__((opencl_global)) dataT *Dest,
                                     __attribute__((opencl_local)) dataT *Src,
                                     size_t NumElements, size_t Stride,
                                     __ocl_event_t E) noexcept {
  for (int i = 0; i < NumElements; i++) {
    Dest[i * Stride] = Src[i];
  }

  return E;
}
#else
template <typename dataT>
extern SYCL_EXTERNAL __ocl_event_t __spirv_GroupAsyncCopy(
    __spv::Scope Execution, __attribute__((opencl_local)) dataT *Dest,
    __attribute__((opencl_global)) dataT *Src, size_t NumElements,
    size_t Stride, __ocl_event_t E) noexcept;

template <typename dataT>
extern SYCL_EXTERNAL __ocl_event_t __spirv_GroupAsyncCopy(
    __spv::Scope Execution, __attribute__((opencl_global)) dataT *Dest,
    __attribute__((opencl_local)) dataT *Src, size_t NumElements, size_t Stride,
    __ocl_event_t E) noexcept;
#endif

#define OpGroupAsyncCopyGlobalToLocal __spirv_GroupAsyncCopy
#define OpGroupAsyncCopyLocalToGlobal __spirv_GroupAsyncCopy

// Atomic SPIR-V builtins
#define __SPIRV_ATOMIC_LOAD(AS, Type)                                          \
  extern SYCL_EXTERNAL Type __spirv_AtomicLoad(                                \
      AS const Type *P, __spv::Scope S, __spv::MemorySemanticsMask O);
#define __SPIRV_ATOMIC_STORE(AS, Type)                                         \
  extern SYCL_EXTERNAL void __spirv_AtomicStore(                               \
      AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask O, Type V);
#define __SPIRV_ATOMIC_EXCHANGE(AS, Type)                                      \
  extern SYCL_EXTERNAL Type __spirv_AtomicExchange(                            \
      AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask O, Type V);
#define __SPIRV_ATOMIC_CMP_EXCHANGE(AS, Type)                                  \
  extern SYCL_EXTERNAL Type __spirv_AtomicCompareExchange(                     \
      AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask E,                \
      __spv::MemorySemanticsMask U, Type V, Type C);
#define __SPIRV_ATOMIC_IADD(AS, Type)                                          \
  extern SYCL_EXTERNAL Type __spirv_AtomicIAdd(                                \
      AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask O, Type V);
#define __SPIRV_ATOMIC_ISUB(AS, Type)                                          \
  extern SYCL_EXTERNAL Type __spirv_AtomicISub(                                \
      AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask O, Type V);
#define __SPIRV_ATOMIC_SMIN(AS, Type)                                          \
  extern SYCL_EXTERNAL Type __spirv_AtomicSMin(                                \
      AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask O, Type V);
#define __SPIRV_ATOMIC_UMIN(AS, Type)                                          \
  extern SYCL_EXTERNAL Type __spirv_AtomicUMin(                                \
      AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask O, Type V);
#define __SPIRV_ATOMIC_SMAX(AS, Type)                                          \
  extern SYCL_EXTERNAL Type __spirv_AtomicSMax(                                \
      AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask O, Type V);
#define __SPIRV_ATOMIC_UMAX(AS, Type)                                          \
  extern SYCL_EXTERNAL Type __spirv_AtomicUMax(                                \
      AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask O, Type V);
#define __SPIRV_ATOMIC_AND(AS, Type)                                           \
  extern SYCL_EXTERNAL Type __spirv_AtomicAnd(                                 \
      AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask O, Type V);
#define __SPIRV_ATOMIC_OR(AS, Type)                                            \
  extern SYCL_EXTERNAL Type __spirv_AtomicOr(                                  \
      AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask O, Type V);
#define __SPIRV_ATOMIC_XOR(AS, Type)                                           \
  extern SYCL_EXTERNAL Type __spirv_AtomicXor(                                 \
      AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask O, Type V);

#define __SPIRV_ATOMIC_FLOAT(AS, Type)                                         \
  __SPIRV_ATOMIC_LOAD(AS, Type)                                                \
  __SPIRV_ATOMIC_STORE(AS, Type)                                               \
  __SPIRV_ATOMIC_EXCHANGE(AS, Type)

#define __SPIRV_ATOMIC_BASE(AS, Type)                                          \
  __SPIRV_ATOMIC_FLOAT(AS, Type)                                               \
  __SPIRV_ATOMIC_CMP_EXCHANGE(AS, Type)                                        \
  __SPIRV_ATOMIC_IADD(AS, Type)                                                \
  __SPIRV_ATOMIC_ISUB(AS, Type)                                                \
  __SPIRV_ATOMIC_AND(AS, Type)                                                 \
  __SPIRV_ATOMIC_OR(AS, Type)                                                  \
  __SPIRV_ATOMIC_XOR(AS, Type)

#define __SPIRV_ATOMIC_SIGNED(AS, Type)                                        \
  __SPIRV_ATOMIC_BASE(AS, Type)                                                \
  __SPIRV_ATOMIC_SMIN(AS, Type)                                                \
  __SPIRV_ATOMIC_SMAX(AS, Type)

#define __SPIRV_ATOMIC_UNSIGNED(AS, Type)                                      \
  __SPIRV_ATOMIC_BASE(AS, Type)                                                \
  __SPIRV_ATOMIC_UMIN(AS, Type)                                                \
  __SPIRV_ATOMIC_UMAX(AS, Type)

// Helper atomic operations which select correct signed/unsigned version
// of atomic min/max based on the signed-ness of the type
#define __SPIRV_ATOMIC_MINMAX(AS, Op)                                          \
  template <typename T>                                                        \
  typename std::enable_if<std::is_signed<T>::value, T>::type                   \
      __spirv_Atomic##Op(AS T *Ptr, __spv::Scope Memory,                       \
                         __spv::MemorySemanticsMask Semantics, T Value) {      \
    return __spirv_AtomicS##Op(Ptr, Memory, Semantics, Value);                 \
  }                                                                            \
  template <typename T>                                                        \
  typename std::enable_if<!std::is_signed<T>::value, T>::type                  \
      __spirv_Atomic##Op(AS T *Ptr, __spv::Scope Memory,                       \
                         __spv::MemorySemanticsMask Semantics, T Value) {      \
    return __spirv_AtomicU##Op(Ptr, Memory, Semantics, Value);                 \
  }

#define __SPIRV_ATOMICS(macro, Arg)                                            \
  macro(__attribute__((opencl_global)), Arg)                                   \
      macro(__attribute__((opencl_local)), Arg)

__SPIRV_ATOMICS(__SPIRV_ATOMIC_FLOAT, float)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_SIGNED, int)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_SIGNED, long)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_SIGNED, long long)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_UNSIGNED, unsigned int)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_UNSIGNED, unsigned long)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_UNSIGNED, unsigned long long)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_MINMAX, Min)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_MINMAX, Max)

extern SYCL_EXTERNAL bool __spirv_GroupAll(__spv::Scope Execution,
                                           bool Predicate) noexcept;

extern SYCL_EXTERNAL bool __spirv_GroupAny(__spv::Scope Execution,
                                           bool Predicate) noexcept;

template <typename dataT>
extern SYCL_EXTERNAL dataT __spirv_GroupBroadcast(__spv::Scope Execution,
                                                  dataT Value,
                                                  uint32_t LocalId) noexcept;

template <typename dataT>
extern SYCL_EXTERNAL dataT
__spirv_GroupBroadcast(__spv::Scope Execution, dataT Value,
                       __ocl_vec_t<size_t, 2> LocalId) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL dataT
__spirv_GroupBroadcast(__spv::Scope Execution, dataT Value,
                       __ocl_vec_t<size_t, 3> LocalId) noexcept;

template <typename dataT>
extern SYCL_EXTERNAL dataT __spirv_GroupIAdd(__spv::Scope Execution,
                                             __spv::GroupOperation Op,
                                             dataT Value) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL dataT __spirv_GroupFAdd(__spv::Scope Execution,
                                             __spv::GroupOperation Op,
                                             dataT Value) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL dataT __spirv_GroupUMin(__spv::Scope Execution,
                                             __spv::GroupOperation Op,
                                             dataT Value) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL dataT __spirv_GroupSMin(__spv::Scope Execution,
                                             __spv::GroupOperation Op,
                                             dataT Value) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL dataT __spirv_GroupFMin(__spv::Scope Execution,
                                             __spv::GroupOperation Op,
                                             dataT Value) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL dataT __spirv_GroupUMax(__spv::Scope Execution,
                                             __spv::GroupOperation Op,
                                             dataT Value) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL dataT __spirv_GroupSMax(__spv::Scope Execution,
                                             __spv::GroupOperation Op,
                                             dataT Value) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL dataT __spirv_GroupFMax(__spv::Scope Execution,
                                             __spv::GroupOperation Op,
                                             dataT Value) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL dataT
__spirv_SubgroupShuffleINTEL(dataT Data, uint32_t InvocationId) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL dataT __spirv_SubgroupShuffleDownINTEL(
    dataT Current, dataT Next, uint32_t Delta) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL dataT __spirv_SubgroupShuffleUpINTEL(
    dataT Previous, dataT Current, uint32_t Delta) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL dataT
__spirv_SubgroupShuffleXorINTEL(dataT Data, uint32_t Value) noexcept;

template <typename dataT>
extern SYCL_EXTERNAL dataT __spirv_SubgroupBlockReadINTEL(
    const __attribute__((opencl_global)) uint8_t *Ptr) noexcept;

template <typename dataT>
extern SYCL_EXTERNAL void
__spirv_SubgroupBlockWriteINTEL(__attribute__((opencl_global)) uint8_t *Ptr,
                                dataT Data) noexcept;

template <typename dataT>
extern SYCL_EXTERNAL dataT __spirv_SubgroupBlockReadINTEL(
    const __attribute__((opencl_global)) uint16_t *Ptr) noexcept;

template <typename dataT>
extern SYCL_EXTERNAL void
__spirv_SubgroupBlockWriteINTEL(__attribute__((opencl_global)) uint16_t *Ptr,
                                dataT Data) noexcept;

template <typename dataT>
extern SYCL_EXTERNAL dataT __spirv_SubgroupBlockReadINTEL(
    const __attribute__((opencl_global)) uint32_t *Ptr) noexcept;

template <typename dataT>
extern SYCL_EXTERNAL void
__spirv_SubgroupBlockWriteINTEL(__attribute__((opencl_global)) uint32_t *Ptr,
                                dataT Data) noexcept;

template <typename dataT>
extern SYCL_EXTERNAL int32_t __spirv_ReadPipe(RPipeTy<dataT> Pipe, dataT *Data,
                                              int32_t Size,
                                              int32_t Alignment) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL int32_t __spirv_WritePipe(WPipeTy<dataT> Pipe,
                                               const dataT *Data, int32_t Size,
                                               int32_t Alignment) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL void
__spirv_ReadPipeBlockingINTEL(RPipeTy<dataT> Pipe, dataT *Data, int32_t Size,
                              int32_t Alignment) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL void
__spirv_WritePipeBlockingINTEL(WPipeTy<dataT> Pipe, const dataT *Data,
                               int32_t Size, int32_t Alignment) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL RPipeTy<dataT> __spirv_CreatePipeFromPipeStorage_read(
    const ConstantPipeStorage *Storage) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL WPipeTy<dataT> __spirv_CreatePipeFromPipeStorage_write(
    const ConstantPipeStorage *Storage) noexcept;

extern SYCL_EXTERNAL void
__spirv_ocl_prefetch(const __attribute__((opencl_global)) char *Ptr,
                     size_t NumBytes) noexcept;

extern SYCL_EXTERNAL int
__spirv_ocl_printf(const __attribute__((opencl_constant)) char *fmt, ...);

#define __SPIRV_COMPARISON(Order, Cmp)                                         \
  template <typename RetT, typename T>                                         \
  extern SYCL_EXTERNAL RetT __spirv_F##Order##Cmp(T, T);

#define __SPIRV_ALL_COMPARISON(Order)                                          \
  __SPIRV_COMPARISON(Order, Equal)                                             \
  __SPIRV_COMPARISON(Order, NotEqual)                                          \
  __SPIRV_COMPARISON(Order, LessThan)                                          \
  __SPIRV_COMPARISON(Order, GreaterThan)                                       \
  __SPIRV_COMPARISON(Order, LessThanEqual)                                     \
  __SPIRV_COMPARISON(Order, GreaterThanEqual)

__SPIRV_ALL_COMPARISON(Unord)
__SPIRV_ALL_COMPARISON(Ord)

#undef __SPIRV_COMPARISON
#undef __SPIRV_ALL_COMPARISON

#define __SPIRV_COMPARISON(Cmp)                                                \
  template <typename RetT, typename T>                                         \
  extern SYCL_EXTERNAL RetT __spirv_##Cmp(T, T);

__SPIRV_COMPARISON(IEqual)
__SPIRV_COMPARISON(INotEqual)

__SPIRV_COMPARISON(ULessThan)
__SPIRV_COMPARISON(UGreaterThanEqual)
__SPIRV_COMPARISON(ULessThanEqual)
__SPIRV_COMPARISON(UGreaterThan)

__SPIRV_COMPARISON(SLessThan)
__SPIRV_COMPARISON(SGreaterThanEqual)
__SPIRV_COMPARISON(SLessThanEqual)
__SPIRV_COMPARISON(SGreaterThan)

__SPIRV_COMPARISON(LessOrGreater)

#undef __SPIRV_COMPARISON

template <typename RetT, typename T> extern SYCL_EXTERNAL RetT __spirv_Any(T);

template <typename RetT, typename T> extern SYCL_EXTERNAL RetT __spirv_All(T);

template <typename RetT, typename T>
extern SYCL_EXTERNAL RetT __spirv_IsFinite(T);

template <typename RetT, typename T> extern SYCL_EXTERNAL RetT __spirv_IsInf(T);

template <typename RetT, typename T> extern SYCL_EXTERNAL RetT __spirv_IsNan(T);

template <typename RetT, typename T>
extern SYCL_EXTERNAL RetT __spirv_IsNormal(T);

template <typename RetT, typename T>
extern SYCL_EXTERNAL RetT __spirv_SignBitSet(T);

template <typename RetT, typename T>
extern SYCL_EXTERNAL RetT __spirv_Ordered(T, T);

template <typename RetT, typename T>
extern SYCL_EXTERNAL RetT __spirv_Unordered(T, T);

template <typename RetT, typename T>
extern SYCL_EXTERNAL RetT __spirv_Dot(T, T);

template <typename T> extern SYCL_EXTERNAL T __spirv_FMul(T, T);

#define __SPIRV_DECLARE_OCL1(name)                                             \
  template <typename RetT, typename T>                                         \
  extern SYCL_EXTERNAL RetT __spirv_ocl_##name(T);

#define __SPIRV_DECLARE_OCL2(name)                                             \
  template <typename RetT, typename T1, typename T2>                           \
  extern SYCL_EXTERNAL RetT __spirv_ocl_##name(T1, T2);

#define __SPIRV_DECLARE_OCL3(name)                                             \
  template <typename RetT, typename T1, typename T2, typename T3>              \
  extern SYCL_EXTERNAL RetT __spirv_ocl_##name(T1, T2, T3);

__SPIRV_DECLARE_OCL1(acos)
__SPIRV_DECLARE_OCL1(acosh)
__SPIRV_DECLARE_OCL1(acospi)
__SPIRV_DECLARE_OCL1(asin)
__SPIRV_DECLARE_OCL1(asinh)
__SPIRV_DECLARE_OCL1(asinpi)
__SPIRV_DECLARE_OCL1(atan)
__SPIRV_DECLARE_OCL2(atan2)
__SPIRV_DECLARE_OCL1(atanh)
__SPIRV_DECLARE_OCL1(atanpi)
__SPIRV_DECLARE_OCL2(atan2pi)
__SPIRV_DECLARE_OCL1(cbrt)
__SPIRV_DECLARE_OCL1(ceil)
__SPIRV_DECLARE_OCL2(copysign)
__SPIRV_DECLARE_OCL1(cos)
__SPIRV_DECLARE_OCL1(cosh)
__SPIRV_DECLARE_OCL1(cospi)
__SPIRV_DECLARE_OCL1(erfc)
__SPIRV_DECLARE_OCL1(erf)
__SPIRV_DECLARE_OCL1(exp)
__SPIRV_DECLARE_OCL1(exp2)
__SPIRV_DECLARE_OCL1(exp10)
__SPIRV_DECLARE_OCL1(expm1)
__SPIRV_DECLARE_OCL1(fabs)
__SPIRV_DECLARE_OCL2(fdim)
__SPIRV_DECLARE_OCL1(floor)
__SPIRV_DECLARE_OCL3(fma)
__SPIRV_DECLARE_OCL2(fmax)
__SPIRV_DECLARE_OCL2(fmin)
__SPIRV_DECLARE_OCL2(fmod)
__SPIRV_DECLARE_OCL2(fract)
__SPIRV_DECLARE_OCL2(frexp)
__SPIRV_DECLARE_OCL2(hypot)
__SPIRV_DECLARE_OCL1(ilogb)
__SPIRV_DECLARE_OCL2(ldexp)
__SPIRV_DECLARE_OCL1(lgamma)
__SPIRV_DECLARE_OCL2(lgamma_r)
__SPIRV_DECLARE_OCL1(log)
__SPIRV_DECLARE_OCL1(log2)
__SPIRV_DECLARE_OCL1(log10)
__SPIRV_DECLARE_OCL1(log1p)
__SPIRV_DECLARE_OCL1(logb)
__SPIRV_DECLARE_OCL3(mad)
__SPIRV_DECLARE_OCL2(maxmag)
__SPIRV_DECLARE_OCL2(minmag)
__SPIRV_DECLARE_OCL2(modf)
__SPIRV_DECLARE_OCL1(nan)
__SPIRV_DECLARE_OCL2(nextafter)
__SPIRV_DECLARE_OCL2(pow)
__SPIRV_DECLARE_OCL2(pown)
__SPIRV_DECLARE_OCL2(powr)
__SPIRV_DECLARE_OCL2(remainder)
__SPIRV_DECLARE_OCL3(remquo)
__SPIRV_DECLARE_OCL1(rint)
__SPIRV_DECLARE_OCL2(rootn)
__SPIRV_DECLARE_OCL1(round)
__SPIRV_DECLARE_OCL1(rsqrt)
__SPIRV_DECLARE_OCL1(sin)
__SPIRV_DECLARE_OCL2(sincos)
__SPIRV_DECLARE_OCL1(sinh)
__SPIRV_DECLARE_OCL1(sinpi)
__SPIRV_DECLARE_OCL1(sqrt)
__SPIRV_DECLARE_OCL1(tan)
__SPIRV_DECLARE_OCL1(tanh)
__SPIRV_DECLARE_OCL1(tanpi)
__SPIRV_DECLARE_OCL1(tgamma)
__SPIRV_DECLARE_OCL1(trunc)
__SPIRV_DECLARE_OCL1(native_cos)
__SPIRV_DECLARE_OCL2(native_divide)
__SPIRV_DECLARE_OCL1(native_exp)
__SPIRV_DECLARE_OCL1(native_exp2)
__SPIRV_DECLARE_OCL1(native_exp10)
__SPIRV_DECLARE_OCL1(native_log)
__SPIRV_DECLARE_OCL1(native_log2)
__SPIRV_DECLARE_OCL1(native_log10)
__SPIRV_DECLARE_OCL2(native_powr)
__SPIRV_DECLARE_OCL1(native_recip)
__SPIRV_DECLARE_OCL1(native_rsqrt)
__SPIRV_DECLARE_OCL1(native_sin)
__SPIRV_DECLARE_OCL1(native_sqrt)
__SPIRV_DECLARE_OCL1(native_tan)
__SPIRV_DECLARE_OCL1(half_cos)
__SPIRV_DECLARE_OCL2(half_divide)
__SPIRV_DECLARE_OCL1(half_exp)
__SPIRV_DECLARE_OCL1(half_exp2)
__SPIRV_DECLARE_OCL1(half_exp10)
__SPIRV_DECLARE_OCL1(half_log)
__SPIRV_DECLARE_OCL1(half_log2)
__SPIRV_DECLARE_OCL1(half_log10)
__SPIRV_DECLARE_OCL2(half_powr)
__SPIRV_DECLARE_OCL1(half_recip)
__SPIRV_DECLARE_OCL1(half_rsqrt)
__SPIRV_DECLARE_OCL1(half_sin)
__SPIRV_DECLARE_OCL1(half_sqrt)
__SPIRV_DECLARE_OCL1(half_tan)
__SPIRV_DECLARE_OCL1(s_abs)
__SPIRV_DECLARE_OCL1(u_abs)
__SPIRV_DECLARE_OCL2(s_abs_diff)
__SPIRV_DECLARE_OCL2(u_abs_diff)
__SPIRV_DECLARE_OCL2(s_add_sat)
__SPIRV_DECLARE_OCL2(u_add_sat)
__SPIRV_DECLARE_OCL2(s_hadd)
__SPIRV_DECLARE_OCL2(u_hadd)
__SPIRV_DECLARE_OCL2(s_rhadd)
__SPIRV_DECLARE_OCL2(u_rhadd)
__SPIRV_DECLARE_OCL3(s_clamp)
__SPIRV_DECLARE_OCL3(u_clamp)
__SPIRV_DECLARE_OCL1(clz)
__SPIRV_DECLARE_OCL1(ctz)
__SPIRV_DECLARE_OCL3(s_mad_hi)
__SPIRV_DECLARE_OCL3(u_mad_hi)
__SPIRV_DECLARE_OCL3(u_mad_sat)
__SPIRV_DECLARE_OCL3(s_mad_sat)
__SPIRV_DECLARE_OCL2(s_max)
__SPIRV_DECLARE_OCL2(u_max)
__SPIRV_DECLARE_OCL2(s_min)
__SPIRV_DECLARE_OCL2(u_min)
__SPIRV_DECLARE_OCL2(s_mul_hi)
__SPIRV_DECLARE_OCL2(u_mul_hi)
__SPIRV_DECLARE_OCL2(rotate)
__SPIRV_DECLARE_OCL2(s_sub_sat)
__SPIRV_DECLARE_OCL2(u_sub_sat)
__SPIRV_DECLARE_OCL2(u_upsample)
__SPIRV_DECLARE_OCL2(s_upsample)
__SPIRV_DECLARE_OCL1(popcount)
__SPIRV_DECLARE_OCL3(s_mad24)
__SPIRV_DECLARE_OCL3(u_mad24)
__SPIRV_DECLARE_OCL2(s_mul24)
__SPIRV_DECLARE_OCL2(u_mul24)
__SPIRV_DECLARE_OCL3(fclamp)
__SPIRV_DECLARE_OCL1(degrees)
__SPIRV_DECLARE_OCL2(fmax_common)
__SPIRV_DECLARE_OCL2(fmin_common)
__SPIRV_DECLARE_OCL3(mix)
__SPIRV_DECLARE_OCL1(radians)
__SPIRV_DECLARE_OCL2(step)
__SPIRV_DECLARE_OCL3(smoothstep)
__SPIRV_DECLARE_OCL1(sign)
__SPIRV_DECLARE_OCL2(cross)
__SPIRV_DECLARE_OCL2(distance)
__SPIRV_DECLARE_OCL1(length)
__SPIRV_DECLARE_OCL1(normalize)
__SPIRV_DECLARE_OCL2(fast_distance)
__SPIRV_DECLARE_OCL1(fast_length)
__SPIRV_DECLARE_OCL1(fast_normalize)
__SPIRV_DECLARE_OCL3(bitselect)
__SPIRV_DECLARE_OCL3(select) // select

#undef __SPIRV_DECLARE_OCL1
#undef __SPIRV_DECLARE_OCL2
#undef __SPIRV_DECLARE_OCL3

#else // if !__SYCL_DEVICE_ONLY__

template <typename dataT>
extern __ocl_event_t
OpGroupAsyncCopyGlobalToLocal(__spv::Scope Execution, dataT *Dest, dataT *Src,
                              size_t NumElements, size_t Stride,
                              __ocl_event_t E) noexcept {
  for (size_t i = 0; i < NumElements; i++) {
    Dest[i] = Src[i * Stride];
  }
  // A real instance of the class is not needed, return dummy pointer.
  return nullptr;
}

template <typename dataT>
extern __ocl_event_t
OpGroupAsyncCopyLocalToGlobal(__spv::Scope Execution, dataT *Dest, dataT *Src,
                              size_t NumElements, size_t Stride,
                              __ocl_event_t E) noexcept {
  for (size_t i = 0; i < NumElements; i++) {
    Dest[i * Stride] = Src[i];
  }
  // A real instance of the class is not needed, return dummy pointer.
  return nullptr;
}

extern void __spirv_ocl_prefetch(const char *Ptr, size_t NumBytes) noexcept;

#endif // !__SYCL_DEVICE_ONLY__

extern SYCL_EXTERNAL void __spirv_ControlBarrier(__spv::Scope Execution,
                                                 __spv::Scope Memory,
                                                 uint32_t Semantics) noexcept;

extern SYCL_EXTERNAL void __spirv_MemoryBarrier(__spv::Scope Memory,
                                                uint32_t Semantics) noexcept;

extern SYCL_EXTERNAL void
__spirv_GroupWaitEvents(__spv::Scope Execution, uint32_t NumEvents,
                        __ocl_event_t *WaitEvents) noexcept;
