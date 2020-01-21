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

template <typename ReTTT, typename ImageT>
extern SYCL_EXTERNAL ReTTT __spirv_ImageQueryFormat(ImageT);

template <typename ReTTT, typename ImageT>
extern SYCL_EXTERNAL ReTTT __spirv_ImageQueryOrder(ImageT);

template <typename ReTTT, typename ImageT>
extern SYCL_EXTERNAL ReTTT __spirv_ImageQuerySize(ImageT);

template <typename ImageT, typename CoordT, typename ValT>
extern SYCL_EXTERNAL void __spirv_ImageWrite(ImageT, CoordT, ValT);

template <class ReTTT, typename ImageT, typename TempArgT>
extern SYCL_EXTERNAL ReTTT __spirv_ImageRead(ImageT, TempArgT);

template <typename ImageT, typename SampledType>
extern SYCL_EXTERNAL SampledType __spirv_SampledImage(ImageT, __ocl_sampler_t);

template <typename SampledType, typename TempRetT, typename TempArgT>
extern SYCL_EXTERNAL TempRetT __spirv_ImageSampleExplicitLod(SampledType,
                                                             TempArgT, int,
                                                             float);

template <typename dataT>
extern SYCL_EXTERNAL __ocl_event_t
__spirv_GroupAsyncCopy(__spv::Scope Execution, __attribute__((opencl_local)) dataT *Dest,
                       __attribute__((opencl_global)) dataT *Src, size_t NumElements, size_t Stride,
                       __ocl_event_t E) noexcept;

template <typename dataT>
extern SYCL_EXTERNAL __ocl_event_t
__spirv_GroupAsyncCopy(__spv::Scope Execution, __attribute__((opencl_global)) dataT *Dest,
                       __attribute__((opencl_local)) dataT *Src, size_t NumElements, size_t Stride,
                       __ocl_event_t E) noexcept;

#define OpGroupAsyncCopyGlobalToLocal __spirv_GroupAsyncCopy
#define OpGroupAsyncCopyLocalToGlobal __spirv_GroupAsyncCopy

// Atomic SPIR-V builtins
#define __SPIRV_ATOMIC_LOAD(AS, Type)                                          \
  extern SYCL_EXTERNAL Type __spirv_AtomicLoad(AS const Type *P,               \
                                               __spv::Scope S,                 \
                                               __spv::MemorySemanticsMask O);
#define __SPIRV_ATOMIC_STORE(AS, Type)                                         \
  extern SYCL_EXTERNAL void __spirv_AtomicStore(AS Type *P, __spv::Scope S,    \
                                                __spv::MemorySemanticsMask O,  \
                                                Type V);
#define __SPIRV_ATOMIC_EXCHANGE(AS, Type)                                      \
  extern SYCL_EXTERNAL Type                                                    \
  __spirv_AtomicExchange(AS Type *P, __spv::Scope S,                           \
                         __spv::MemorySemanticsMask O, Type V);
#define __SPIRV_ATOMIC_CMP_EXCHANGE(AS, Type)                                  \
  extern SYCL_EXTERNAL Type                                                    \
  __spirv_AtomicCompareExchange(AS Type *P, __spv::Scope S,                    \
                                __spv::MemorySemanticsMask E,                  \
                                __spv::MemorySemanticsMask U, Type V, Type C);
#define __SPIRV_ATOMIC_IADD(AS, Type)                                          \
  extern SYCL_EXTERNAL Type                                                    \
  __spirv_AtomicIAdd(AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask O, \
                     Type V);
#define __SPIRV_ATOMIC_ISUB(AS, Type)                                          \
  extern SYCL_EXTERNAL Type                                                    \
  __spirv_AtomicISub(AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask O, \
                     Type V);
#define __SPIRV_ATOMIC_SMIN(AS, Type)                                          \
  extern SYCL_EXTERNAL Type                                                    \
  __spirv_AtomicSMin(AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask O, \
                     Type V);
#define __SPIRV_ATOMIC_UMIN(AS, Type)                                          \
  extern SYCL_EXTERNAL Type                                                    \
  __spirv_AtomicUMin(AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask O, \
                     Type V);
#define __SPIRV_ATOMIC_SMAX(AS, Type)                                          \
  extern SYCL_EXTERNAL Type                                                    \
  __spirv_AtomicSMax(AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask O, \
                     Type V);
#define __SPIRV_ATOMIC_UMAX(AS, Type)                                          \
  extern SYCL_EXTERNAL Type                                                    \
  __spirv_AtomicUMax(AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask O, \
                     Type V);
#define __SPIRV_ATOMIC_AND(AS, Type)                                           \
  extern SYCL_EXTERNAL Type                                                    \
  __spirv_AtomicAnd(AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask O,  \
                    Type V);
#define __SPIRV_ATOMIC_OR(AS, Type)                                            \
  extern SYCL_EXTERNAL Type                                                    \
  __spirv_AtomicOr(AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask O,   \
                   Type V);
#define __SPIRV_ATOMIC_XOR(AS, Type)                                           \
  extern SYCL_EXTERNAL Type                                                    \
  __spirv_AtomicXor(AS Type *P, __spv::Scope S, __spv::MemorySemanticsMask O,  \
                    Type V);

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

#define __SPIRV_ATOMICS(macro, Arg) macro(__attribute__((opencl_global)), Arg) macro(__attribute__((opencl_local)), Arg)

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
extern SYCL_EXTERNAL dataT
__spirv_SubgroupShuffleDownINTEL(dataT Current, dataT Next, uint32_t Delta) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL dataT
__spirv_SubgroupShuffleUpINTEL(dataT Previous, dataT Current, uint32_t Delta) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL dataT
__spirv_SubgroupShuffleXorINTEL(dataT Data, uint32_t Value) noexcept;

template <typename dataT>
extern SYCL_EXTERNAL dataT
__spirv_SubgroupBlockReadINTEL(const __attribute__((opencl_global))
                               uint8_t *Ptr) noexcept;

template <typename dataT>
extern SYCL_EXTERNAL void
__spirv_SubgroupBlockWriteINTEL(__attribute__((opencl_global)) uint8_t *Ptr,
                                dataT Data) noexcept;

template <typename dataT>
extern SYCL_EXTERNAL dataT
__spirv_SubgroupBlockReadINTEL(const __attribute__((opencl_global)) uint16_t *Ptr) noexcept;

template <typename dataT>
extern SYCL_EXTERNAL void
__spirv_SubgroupBlockWriteINTEL(__attribute__((opencl_global)) uint16_t *Ptr,
                                dataT Data) noexcept;

template <typename dataT>
extern SYCL_EXTERNAL dataT
__spirv_SubgroupBlockReadINTEL(const __attribute__((opencl_global)) uint32_t *Ptr) noexcept;

template <typename dataT>
extern SYCL_EXTERNAL void
__spirv_SubgroupBlockWriteINTEL(__attribute__((opencl_global)) uint32_t *Ptr,
                                dataT Data) noexcept;

template <typename dataT>
extern SYCL_EXTERNAL int32_t
__spirv_ReadPipe(RPipeTy<dataT> Pipe, dataT *Data, int32_t Size,
                 int32_t Alignment) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL int32_t
__spirv_WritePipe(WPipeTy<dataT> Pipe, const dataT *Data, int32_t Size,
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

extern SYCL_EXTERNAL
int __spirv_ocl_printf(const __attribute__((opencl_constant)) char *fmt, ...);

#define COMPARISON(Order, Cmp)                                                 \
  template <typename ReTTT, typename T>                                        \
  extern SYCL_EXTERNAL                                                         \
  ReTTT __spirv_F##Order##Cmp(T, T);

#define ALL_COMPARISON(Order)                                                  \
  COMPARISON(Order, Equal)                                                     \
  COMPARISON(Order, NotEqual)                                                  \
  COMPARISON(Order, LessThan)                                                  \
  COMPARISON(Order, GreaterThan)                                               \
  COMPARISON(Order, LessThanEqual)                                             \
  COMPARISON(Order, GreaterThanEqual)

ALL_COMPARISON(Unord)
ALL_COMPARISON(Ord)

#undef COMPARISON
#undef ALL_COMPARISON

#define COMPARISON(Cmp)                                                        \
  template <typename ReTTT, typename T>                                        \
  extern SYCL_EXTERNAL                                                         \
  ReTTT __spirv_##Cmp(T, T);

COMPARISON(IEqual)
COMPARISON(INotEqual)

COMPARISON(ULessThan)
COMPARISON(UGreaterThanEqual)
COMPARISON(ULessThanEqual)
COMPARISON(UGreaterThan)

COMPARISON(SLessThan)
COMPARISON(SGreaterThanEqual)
COMPARISON(SLessThanEqual)
COMPARISON(SGreaterThan)

COMPARISON(LessOrGreater)

#undef COMPARISON

template <typename ReTTT, typename T>
extern SYCL_EXTERNAL
ReTTT __spirv_Any(T);

template <typename ReTTT, typename T>
extern SYCL_EXTERNAL
ReTTT __spirv_All(T);

template <typename ReTTT, typename T>
extern SYCL_EXTERNAL
ReTTT __spirv_IsFinite(T);

template <typename ReTTT, typename T>
extern SYCL_EXTERNAL
ReTTT __spirv_IsInf(T);

template <typename ReTTT, typename T>
extern SYCL_EXTERNAL
ReTTT __spirv_IsNan(T);

template <typename ReTTT, typename T>
extern SYCL_EXTERNAL
ReTTT __spirv_IsNormal(T);

template <typename ReTTT, typename T>
extern SYCL_EXTERNAL
ReTTT __spirv_SignBitSet(T);

template <typename ReTTT, typename T>
extern SYCL_EXTERNAL
ReTTT __spirv_Ordered(T, T);

template <typename ReTTT, typename T>
extern SYCL_EXTERNAL
ReTTT __spirv_Unordered(T, T);

template <typename ReTTT, typename T>
extern SYCL_EXTERNAL
ReTTT __spirv_Dot(T, T);

template <typename T>
extern SYCL_EXTERNAL
T __spirv_FMul(T, T);

#define DECLARE_OCL1(name)                                                     \
  template <typename ReTTT, typename T>                                        \
  extern SYCL_EXTERNAL                                                         \
  ReTTT __spirv_ocl_##name(T);

#define DECLARE_OCL2(name)                                                     \
  template <typename ReTTT, typename T1, typename T2>                          \
  extern SYCL_EXTERNAL                                                         \
  ReTTT __spirv_ocl_##name(T1, T2);

#define DECLARE_OCL3(name)                                                     \
  template <typename ReTTT, typename T1, typename T2, typename T3>             \
  extern SYCL_EXTERNAL                                                         \
  ReTTT __spirv_ocl_##name(T1, T2, T3);

DECLARE_OCL1(acos)
DECLARE_OCL1(acosh)
DECLARE_OCL1(acospi)
DECLARE_OCL1(asin)
DECLARE_OCL1(asinh)
DECLARE_OCL1(asinpi)
DECLARE_OCL1(atan)
DECLARE_OCL2(atan2)
DECLARE_OCL1(atanh)
DECLARE_OCL1(atanpi)
DECLARE_OCL2(atan2pi)
DECLARE_OCL1(cbrt)
DECLARE_OCL1(ceil)
DECLARE_OCL2(copysign)
DECLARE_OCL1(cos)
DECLARE_OCL1(cosh)
DECLARE_OCL1(cospi)
DECLARE_OCL1(erfc)
DECLARE_OCL1(erf)
DECLARE_OCL1(exp)
DECLARE_OCL1(exp2)
DECLARE_OCL1(exp10)
DECLARE_OCL1(expm1)
DECLARE_OCL1(fabs)
DECLARE_OCL2(fdim)
DECLARE_OCL1(floor)
DECLARE_OCL3(fma)
DECLARE_OCL2(fmax)
DECLARE_OCL2(fmin)
DECLARE_OCL2(fmod)
DECLARE_OCL2(fract)
DECLARE_OCL2(frexp)
DECLARE_OCL2(hypot)
DECLARE_OCL1(ilogb)
DECLARE_OCL2(ldexp)
DECLARE_OCL1(lgamma)
DECLARE_OCL2(lgamma_r)
DECLARE_OCL1(log)
DECLARE_OCL1(log2)
DECLARE_OCL1(log10)
DECLARE_OCL1(log1p)
DECLARE_OCL1(logb)
DECLARE_OCL3(mad)
DECLARE_OCL2(maxmag)
DECLARE_OCL2(minmag)
DECLARE_OCL2(modf)
DECLARE_OCL1(nan)
DECLARE_OCL2(nextafter)
DECLARE_OCL2(pow)
DECLARE_OCL2(pown)
DECLARE_OCL2(powr)
DECLARE_OCL2(remainder)
DECLARE_OCL3(remquo)
DECLARE_OCL1(rint)
DECLARE_OCL2(rootn)
DECLARE_OCL1(round)
DECLARE_OCL1(rsqrt)
DECLARE_OCL1(sin)
DECLARE_OCL2(sincos)
DECLARE_OCL1(sinh)
DECLARE_OCL1(sinpi)
DECLARE_OCL1(sqrt)
DECLARE_OCL1(tan)
DECLARE_OCL1(tanh)
DECLARE_OCL1(tanpi)
DECLARE_OCL1(tgamma)
DECLARE_OCL1(trunc)
DECLARE_OCL1(native_cos)
DECLARE_OCL2(native_divide)
DECLARE_OCL1(native_exp)
DECLARE_OCL1(native_exp2)
DECLARE_OCL1(native_exp10)
DECLARE_OCL1(native_log)
DECLARE_OCL1(native_log2)
DECLARE_OCL1(native_log10)
DECLARE_OCL2(native_powr)
DECLARE_OCL1(native_recip)
DECLARE_OCL1(native_rsqrt)
DECLARE_OCL1(native_sin)
DECLARE_OCL1(native_sqrt)
DECLARE_OCL1(native_tan)
DECLARE_OCL1(half_cos)
DECLARE_OCL2(half_divide)
DECLARE_OCL1(half_exp)
DECLARE_OCL1(half_exp2)
DECLARE_OCL1(half_exp10)
DECLARE_OCL1(half_log)
DECLARE_OCL1(half_log2)
DECLARE_OCL1(half_log10)
DECLARE_OCL2(half_powr)
DECLARE_OCL1(half_recip)
DECLARE_OCL1(half_rsqrt)
DECLARE_OCL1(half_sin)
DECLARE_OCL1(half_sqrt)
DECLARE_OCL1(half_tan)
DECLARE_OCL1(s_abs)
DECLARE_OCL1(u_abs)
DECLARE_OCL2(s_abs_diff)
DECLARE_OCL2(u_abs_diff)
DECLARE_OCL2(s_add_sat)
DECLARE_OCL2(u_add_sat)
DECLARE_OCL2(s_hadd)
DECLARE_OCL2(u_hadd)
DECLARE_OCL2(s_rhadd)
DECLARE_OCL2(u_rhadd)
DECLARE_OCL3(s_clamp)
DECLARE_OCL3(u_clamp)
DECLARE_OCL1(clz)
DECLARE_OCL1(ctz)
DECLARE_OCL3(s_mad_hi)
DECLARE_OCL3(u_mad_hi)
DECLARE_OCL3(u_mad_sat)
DECLARE_OCL3(s_mad_sat)
DECLARE_OCL2(s_max)
DECLARE_OCL2(u_max)
DECLARE_OCL2(s_min)
DECLARE_OCL2(u_min)
DECLARE_OCL2(s_mul_hi)
DECLARE_OCL2(u_mul_hi)
DECLARE_OCL2(rotate)
DECLARE_OCL2(s_sub_sat)
DECLARE_OCL2(u_sub_sat)
DECLARE_OCL2(u_upsample)
DECLARE_OCL2(s_upsample)
DECLARE_OCL1(popcount)
DECLARE_OCL3(s_mad24)
DECLARE_OCL3(u_mad24)
DECLARE_OCL2(s_mul24)
DECLARE_OCL2(u_mul24)
DECLARE_OCL3(fclamp)
DECLARE_OCL1(degrees)
DECLARE_OCL2(fmax_common)
DECLARE_OCL2(fmin_common)
DECLARE_OCL3(mix)
DECLARE_OCL1(radians)
DECLARE_OCL2(step)
DECLARE_OCL3(smoothstep)
DECLARE_OCL1(sign)
DECLARE_OCL2(cross)
DECLARE_OCL2(distance)
DECLARE_OCL1(length)
DECLARE_OCL1(normalize)
DECLARE_OCL2(fast_distance)
DECLARE_OCL1(fast_length)
DECLARE_OCL1(fast_normalize)
DECLARE_OCL3(bitselect)
DECLARE_OCL3(select) // select

#undef DECLARE_OCL1
#undef DECLARE_OCL2
#undef DECLARE_OCL3

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

extern SYCL_EXTERNAL void
__spirv_ControlBarrier(__spv::Scope Execution, __spv::Scope Memory,
                       uint32_t Semantics) noexcept;

extern SYCL_EXTERNAL void
__spirv_MemoryBarrier(__spv::Scope Memory, uint32_t Semantics) noexcept;

extern SYCL_EXTERNAL void
__spirv_GroupWaitEvents(__spv::Scope Execution, uint32_t NumEvents,
                        __ocl_event_t *WaitEvents) noexcept;

