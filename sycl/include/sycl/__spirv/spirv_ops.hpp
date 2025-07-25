//==----------- spirv_ops.hpp --- SPIRV operations -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_types.hpp>       // for Scope, __ocl_event_t
#include <sycl/detail/defines_elementary.hpp> // for __DPCPP_SYCL_EXTERNAL
#include <sycl/detail/export.hpp>             // for __SYCL_EXPORT

#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t
#include <type_traits>
#include <utility> // for pair

// Convergent attribute
#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_CONVERGENT__ __attribute__((convergent))
#else
#define __SYCL_CONVERGENT__
#endif

#ifdef __SYCL_DEVICE_ONLY__

extern __DPCPP_SYCL_EXTERNAL float __spirv_RoundFToTF32INTEL(float a);

template <typename T, typename Tp, std::size_t R, std::size_t C,
          __spv::MatrixUse U,
          __spv::MatrixLayout L = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern __DPCPP_SYCL_EXTERNAL
    __spv::__spirv_CooperativeMatrixKHR<Tp, S, R, C, U> *
    __spirv_CooperativeMatrixLoadKHR(T *Ptr, __spv::MatrixLayout Layout = L,
                                     std::size_t Stride = 0,
                                     int MemOperand = 0);

template <typename T, typename Tp, std::size_t R, std::size_t C,
          __spv::MatrixUse U,
          __spv::MatrixLayout L = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern __DPCPP_SYCL_EXTERNAL void __spirv_CooperativeMatrixStoreKHR(
    T *Ptr, __spv::__spirv_CooperativeMatrixKHR<Tp, S, R, C, U> *Object,
    __spv::MatrixLayout Layout = L, std::size_t Stride = 0, int MemOperand = 0);

template <typename T, std::size_t R, std::size_t C, __spv::MatrixUse U,
          __spv::MatrixLayout L = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern __DPCPP_SYCL_EXTERNAL size_t __spirv_CooperativeMatrixLengthKHR(
    __spv::__spirv_CooperativeMatrixKHR<T, S, R, C, U> *);

template <typename T, typename Tp, std::size_t R, std::size_t C,
          __spv::MatrixUse U,
          __spv::MatrixLayout L = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern __DPCPP_SYCL_EXTERNAL
    __spv::__spirv_CooperativeMatrixKHR<Tp, S, R, C, U> *
    __spirv_CooperativeMatrixConstructCheckedINTEL(const T Value, size_t Height,
                                                   size_t Stride, size_t Width,
                                                   size_t CoordX,
                                                   size_t CoordY);

template <typename T, typename Tp, std::size_t R, std::size_t C,
          __spv::MatrixUse U,
          __spv::MatrixLayout L = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern __DPCPP_SYCL_EXTERNAL
    __spv::__spirv_CooperativeMatrixKHR<Tp, S, R, C, U> *
    __spirv_CooperativeMatrixLoadCheckedINTEL(T *Ptr, std::size_t Stride,
                                              size_t Height, size_t Width,
                                              size_t CoordX, size_t CoordY,
                                              __spv::MatrixLayout Layout = L,
                                              int MemOperand = 0);

template <typename T, typename Tp, std::size_t R, std::size_t C,
          __spv::MatrixUse U,
          __spv::MatrixLayout L = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern __DPCPP_SYCL_EXTERNAL void __spirv_CooperativeMatrixStoreCheckedINTEL(
    T *Ptr, __spv::__spirv_CooperativeMatrixKHR<Tp, S, R, C, U> *Object,
    std::size_t Stride, size_t Height, size_t Width, size_t CoordX,
    size_t CoordY, __spv::MatrixLayout Layout = L, int MemOperand = 0);

template <typename TA, typename TB, typename TC, typename TD, std::size_t M,
          std::size_t K, std::size_t N, __spv::MatrixUse UA,
          __spv::MatrixUse UB, __spv::MatrixUse UC,
          __spv::MatrixLayout LA = __spv::MatrixLayout::RowMajor,
          __spv::MatrixLayout LB = __spv::MatrixLayout::RowMajor,
          __spv::MatrixLayout LC = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern __DPCPP_SYCL_EXTERNAL
    __spv::__spirv_CooperativeMatrixKHR<TD, S, M, N, UC> *
    __spirv_CooperativeMatrixMulAddKHR(
        __spv::__spirv_CooperativeMatrixKHR<TA, S, M, K, UA> *A,
        __spv::__spirv_CooperativeMatrixKHR<TB, S, K, N, UB> *B,
        __spv::__spirv_CooperativeMatrixKHR<TC, S, M, N, UC> *C,
        size_t Operands = 0);

template <typename T, typename Tp, std::size_t R, std::size_t C,
          __spv::MatrixUse U,
          __spv::MatrixLayout L = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern __DPCPP_SYCL_EXTERNAL
    __spv::__spirv_CooperativeMatrixKHR<Tp, S, R, C, U> *
    __spirv_CompositeConstruct(const T v);

// TODO: replace with __spirv_CooperativeMatrixGetElementCoordINTEL when ready
template <typename T, std::size_t R, std::size_t C, __spv::MatrixUse U,
          __spv::MatrixLayout L = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern __DPCPP_SYCL_EXTERNAL __ocl_vec_t<uint32_t, 2>
__spirv_JointMatrixGetElementCoordINTEL(
    __spv::__spirv_CooperativeMatrixKHR<T, S, R, C, U> *, size_t i);

// AccessChain followed by load/store serves to extract/insert and element
// from/to the matrix
template <typename Ts, typename T, std::size_t R, std::size_t C,
          __spv::MatrixUse U,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern __DPCPP_SYCL_EXTERNAL Ts *
__spirv_AccessChain(__spv::__spirv_CooperativeMatrixKHR<T, S, R, C, U> **,
                    size_t i);

template <typename T, typename Tp, std::size_t R, std::size_t C,
          __spv::MatrixUse U,
          __spv::MatrixLayout L = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern __DPCPP_SYCL_EXTERNAL
    __spv::__spirv_CooperativeMatrixKHR<Tp, S, R, C, U> *
    __spirv_CooperativeMatrixConstructCheckedINTEL(int32_t CoordX,
                                                   int32_t CoordY,
                                                   uint32_t Height,
                                                   uint32_t Width,
                                                   const T Value);

template <typename T, typename Tp, std::size_t R, std::size_t C,
          __spv::MatrixUse U,
          __spv::MatrixLayout L = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern __DPCPP_SYCL_EXTERNAL
    __spv::__spirv_CooperativeMatrixKHR<Tp, S, R, C, U> *
    __spirv_CooperativeMatrixLoadCheckedINTEL(
        T *Ptr, int32_t CoordX, int32_t CoordY, __spv::MatrixLayout Layout = L,
        uint32_t Height = 0, uint32_t Width = 0, std::size_t Stride = 0,
        int MemOperand = 0);

template <typename T, typename Tp, std::size_t R, std::size_t C,
          __spv::MatrixUse U,
          __spv::MatrixLayout L = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern __DPCPP_SYCL_EXTERNAL void __spirv_CooperativeMatrixStoreCheckedINTEL(
    T *Ptr, int32_t CoordX, int32_t CoordY,
    __spv::__spirv_CooperativeMatrixKHR<Tp, S, R, C, U> *Object,
    __spv::MatrixLayout Layout = L, uint32_t Height = 0, uint32_t Width = 0,
    std::size_t Stride = 0, int MemOperand = 0);

template <typename T>
extern __DPCPP_SYCL_EXTERNAL void __spirv_CooperativeMatrixPrefetchINTEL(
    T *Ptr, uint32_t NumRows, uint32_t NumCols, unsigned int CacheLevel,
    __spv::MatrixLayout Layout, size_t Stride);

#ifndef __SPIRV_BUILTIN_DECLARATIONS__
#error                                                                         \
    "SPIR-V built-ins are not available. Please set -fdeclare-spirv-builtins flag."
#endif

template <typename RetT, typename ImageT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_ImageQueryFormat(ImageT);

template <typename RetT, typename ImageT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_ImageQueryOrder(ImageT);

template <typename RetT, typename ImageT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_ImageQuerySize(ImageT);

template <typename ImageT, typename CoordT, typename ValT>
extern __DPCPP_SYCL_EXTERNAL void __spirv_ImageWrite(ImageT, CoordT, ValT);

template <class RetT, typename ImageT, typename TempArgT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_ImageRead(ImageT, TempArgT);

template <class RetT, typename ImageT, typename TempArgT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_ImageFetch(ImageT, TempArgT);

template <class RetT, typename ImageT, typename TempArgT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_SampledImageFetch(ImageT, TempArgT);

template <class RetT, typename ImageT, typename TempArgT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_ImageArrayFetch(ImageT, TempArgT,
                                                          int);

template <class RetT, typename ImageT, typename TempArgT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_SampledImageArrayFetch(ImageT,
                                                                 TempArgT, int);

template <class RetT, typename ImageT, typename TempArgT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_SampledImageGather(ImageT, TempArgT,
                                                             unsigned);

template <class RetT, typename ImageT, typename TempArgT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_ImageArrayRead(ImageT, TempArgT, int);

template <typename ImageT, typename CoordT, typename ValT>
extern __DPCPP_SYCL_EXTERNAL void __spirv_ImageArrayWrite(ImageT, CoordT, int,
                                                          ValT);

template <typename ImageT, typename SampledType>
extern __DPCPP_SYCL_EXTERNAL SampledType __spirv_SampledImage(ImageT,
                                                              __ocl_sampler_t);

template <typename SampledType, typename TempRetT, typename TempArgT>
extern __DPCPP_SYCL_EXTERNAL TempRetT
__spirv_ImageSampleExplicitLod(SampledType, TempArgT, int, float);

template <typename SampledType, typename TempRetT, typename TempArgT>
extern __DPCPP_SYCL_EXTERNAL TempRetT
__spirv_ImageSampleExplicitLod(SampledType, TempArgT, int, TempArgT, TempArgT);

template <typename SampledType, typename TempRetT, typename TempArgT>
extern __DPCPP_SYCL_EXTERNAL TempRetT __spirv_ImageSampleCubemap(SampledType,
                                                                 TempArgT);

template <typename RetT, class HandleT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_ConvertHandleToImageINTEL(HandleT);

template <typename RetT, class HandleT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_ConvertHandleToSamplerINTEL(HandleT);

template <typename RetT, class HandleT>
extern __DPCPP_SYCL_EXTERNAL
    RetT __spirv_ConvertHandleToSampledImageINTEL(HandleT);

// Atomic SPIR-V builtins
// TODO: drop these forward-declarations.
// As of now, compiler does not forward-declare long long overloads for
// these and as such we can't drop anything from here. But ideally, we should
// rely on the compiler to generate those - that would allow to drop
// spirv_ops.hpp include from more files.
#define __SPIRV_ATOMIC_LOAD(AS, Type)                                          \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicLoad(AS Type *P, int S,      \
                                                       int O) noexcept;
#define __SPIRV_ATOMIC_STORE(AS, Type)                                         \
  extern __DPCPP_SYCL_EXTERNAL void __spirv_AtomicStore(                       \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_EXCHANGE(AS, Type)                                      \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicExchange(                    \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_CMP_EXCHANGE(AS, Type)                                  \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicCompareExchange(             \
      AS Type *P, int S, int E, int U, Type V, Type C) noexcept;
#define __SPIRV_ATOMIC_IADD(AS, Type)                                          \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicIAdd(                        \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_ISUB(AS, Type)                                          \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicISub(                        \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_FADD(AS, Type)                                          \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicFAddEXT(                     \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_SMIN(AS, Type)                                          \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicSMin(                        \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_UMIN(AS, Type)                                          \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicUMin(                        \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_FMIN(AS, Type)                                          \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicFMinEXT(                     \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_SMAX(AS, Type)                                          \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicSMax(                        \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_UMAX(AS, Type)                                          \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicUMax(                        \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_FMAX(AS, Type)                                          \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicFMaxEXT(                     \
      AS Type *P, int S, int O, Type V) noexcept;
#define __SPIRV_ATOMIC_AND(AS, Type)                                           \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicAnd(AS Type *P, int S,       \
                                                      int O, Type V) noexcept;
#define __SPIRV_ATOMIC_OR(AS, Type)                                            \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicOr(AS Type *P, int S, int O, \
                                                     Type V) noexcept;
#define __SPIRV_ATOMIC_XOR(AS, Type)                                           \
  extern __DPCPP_SYCL_EXTERNAL Type __spirv_AtomicXor(AS Type *P, int S,       \
                                                      int O, Type V) noexcept;

#define __SPIRV_ATOMIC_FLOAT(AS, Type)                                         \
  __SPIRV_ATOMIC_FADD(AS, Type)                                                \
  __SPIRV_ATOMIC_FMIN(AS, Type)                                                \
  __SPIRV_ATOMIC_FMAX(AS, Type)                                                \
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
// of atomic min/max based on the type
#define __SPIRV_ATOMIC_MINMAX(AS, Op)                                          \
  template <typename T>                                                        \
  typename std::enable_if_t<                                                   \
      std::is_integral<T>::value && std::is_signed<T>::value, T>               \
      __spirv_Atomic##Op(AS T *Ptr, int Memory, int Semantics,                 \
                         T Value) noexcept {                                   \
    return __spirv_AtomicS##Op(Ptr, Memory, Semantics, Value);                 \
  }                                                                            \
  template <typename T>                                                        \
  typename std::enable_if_t<                                                   \
      std::is_integral<T>::value && !std::is_signed<T>::value, T>              \
      __spirv_Atomic##Op(AS T *Ptr, int Memory, int Semantics,                 \
                         T Value) noexcept {                                   \
    return __spirv_AtomicU##Op(Ptr, Memory, Semantics, Value);                 \
  }                                                                            \
  template <typename T>                                                        \
  typename std::enable_if_t<std::is_floating_point<T>::value, T>               \
      __spirv_Atomic##Op(AS T *Ptr, int Memory, int Semantics,                 \
                         T Value) noexcept {                                   \
    return __spirv_AtomicF##Op##EXT(Ptr, Memory, Semantics, Value);            \
  }

#define __SPIRV_ATOMICS(macro, Arg)                                            \
  macro(__attribute__((opencl_global)), Arg)                                   \
      macro(__attribute__((opencl_local)), Arg) macro(, Arg)

__SPIRV_ATOMICS(__SPIRV_ATOMIC_FLOAT, _Float16)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_FLOAT, float)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_FLOAT, double)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_SIGNED, int)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_SIGNED, long)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_SIGNED, long long)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_UNSIGNED, unsigned int)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_UNSIGNED, unsigned long)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_UNSIGNED, unsigned long long)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_MINMAX, Min)
__SPIRV_ATOMICS(__SPIRV_ATOMIC_MINMAX, Max)

#undef __SPIRV_ATOMICS
#undef __SPIRV_ATOMIC_AND
#undef __SPIRV_ATOMIC_BASE
#undef __SPIRV_ATOMIC_CMP_EXCHANGE
#undef __SPIRV_ATOMIC_EXCHANGE
#undef __SPIRV_ATOMIC_FADD
#undef __SPIRV_ATOMIC_FLOAT
#undef __SPIRV_ATOMIC_FMAX
#undef __SPIRV_ATOMIC_FMIN
#undef __SPIRV_ATOMIC_IADD
#undef __SPIRV_ATOMIC_ISUB
#undef __SPIRV_ATOMIC_LOAD
#undef __SPIRV_ATOMIC_MINMAX
#undef __SPIRV_ATOMIC_OR
#undef __SPIRV_ATOMIC_SIGNED
#undef __SPIRV_ATOMIC_SMAX
#undef __SPIRV_ATOMIC_SMIN
#undef __SPIRV_ATOMIC_STORE
#undef __SPIRV_ATOMIC_UMAX
#undef __SPIRV_ATOMIC_UMIN
#undef __SPIRV_ATOMIC_UNSIGNED
#undef __SPIRV_ATOMIC_XOR

template <typename dataT>
__SYCL_CONVERGENT__ extern __DPCPP_SYCL_EXTERNAL dataT
__spirv_SubgroupBlockReadINTEL(const __attribute__((opencl_global))
                               uint8_t *Ptr) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern __DPCPP_SYCL_EXTERNAL dataT
__spirv_SubgroupBlockReadINTEL(const __attribute__((opencl_global))
                               uint16_t *Ptr) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern __DPCPP_SYCL_EXTERNAL dataT
__spirv_SubgroupBlockReadINTEL(const __attribute__((opencl_global))
                               uint32_t *Ptr) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern __DPCPP_SYCL_EXTERNAL dataT
__spirv_SubgroupBlockReadINTEL(const __attribute__((opencl_global))
                               uint64_t *Ptr) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern __DPCPP_SYCL_EXTERNAL dataT
__spirv_SubgroupBlockReadINTEL(const __attribute__((opencl_local))
                               uint8_t *Ptr) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern __DPCPP_SYCL_EXTERNAL dataT
__spirv_SubgroupBlockReadINTEL(const __attribute__((opencl_local))
                               uint16_t *Ptr) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern __DPCPP_SYCL_EXTERNAL dataT
__spirv_SubgroupBlockReadINTEL(const __attribute__((opencl_local))
                               uint32_t *Ptr) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern __DPCPP_SYCL_EXTERNAL dataT
__spirv_SubgroupBlockReadINTEL(const __attribute__((opencl_local))
                               uint64_t *Ptr) noexcept;

template <int W, int rW>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<rW>
__spirv_FixedSqrtINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I, int32_t rI,
                       int32_t Quantization = 0, int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<rW>
__spirv_FixedRecipINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I,
                        int32_t rI, int32_t Quantization = 0,
                        int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<rW>
__spirv_FixedRsqrtINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I,
                        int32_t rI, int32_t Quantization = 0,
                        int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<rW>
__spirv_FixedSinINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I, int32_t rI,
                      int32_t Quantization = 0, int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<rW>
__spirv_FixedCosINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I, int32_t rI,
                      int32_t Quantization = 0, int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<2 * rW>
__spirv_FixedSinCosINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I,
                         int32_t rI, int32_t Quantization = 0,
                         int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<rW>
__spirv_FixedSinPiINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I,
                        int32_t rI, int32_t Quantization = 0,
                        int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<rW>
__spirv_FixedCosPiINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I,
                        int32_t rI, int32_t Quantization = 0,
                        int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<2 * rW>
__spirv_FixedSinCosPiINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I,
                           int32_t rI, int32_t Quantization = 0,
                           int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<rW>
__spirv_FixedLogINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I, int32_t rI,
                      int32_t Quantization = 0, int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<rW>
__spirv_FixedExpINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I, int32_t rI,
                      int32_t Quantization = 0, int32_t Overflow = 0) noexcept;

// In the following built-ins width of arbitrary precision integer type for
// a floating point variable should be equal to sum of corresponding
// exponent width E, mantissa width M and 1 for sign bit. I.e. WA = EA + MA + 1.
template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatCastINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatCastFromIntINTEL(sycl::detail::ap_int<WA> A, int32_t Mout,
                                       bool FromSign = false,
                                       int32_t EnableSubnormals = 0,
                                       int32_t RoundingMode = 0,
                                       int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatCastToIntINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                     bool ToSign = false,
                                     int32_t EnableSubnormals = 0,
                                     int32_t RoundingMode = 0,
                                     int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatAddINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                               sycl::detail::ap_int<WB> B, int32_t MB,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatSubINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                               sycl::detail::ap_int<WB> B, int32_t MB,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatMulINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                               sycl::detail::ap_int<WB> B, int32_t MB,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatDivINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                               sycl::detail::ap_int<WB> B, int32_t MB,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

// Comparison built-ins don't use Subnormal Support, Rounding Mode and
// Rounding Accuracy.
template <int WA, int WB>
extern __DPCPP_SYCL_EXTERNAL bool
__spirv_ArbitraryFloatGTINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                              sycl::detail::ap_int<WB> B, int32_t MB) noexcept;

template <int WA, int WB>
extern __DPCPP_SYCL_EXTERNAL bool
__spirv_ArbitraryFloatGEINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                              sycl::detail::ap_int<WB> B, int32_t MB) noexcept;

template <int WA, int WB>
extern __DPCPP_SYCL_EXTERNAL bool
__spirv_ArbitraryFloatLTINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                              sycl::detail::ap_int<WB> B, int32_t MB) noexcept;

template <int WA, int WB>
extern __DPCPP_SYCL_EXTERNAL bool
__spirv_ArbitraryFloatLEINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                              sycl::detail::ap_int<WB> B, int32_t MB) noexcept;

template <int WA, int WB>
extern __DPCPP_SYCL_EXTERNAL bool
__spirv_ArbitraryFloatEQINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                              sycl::detail::ap_int<WB> B, int32_t MB) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatRecipINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatRSqrtINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatCbrtINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatHypotINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                 sycl::detail::ap_int<WB> B, int32_t MB,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatSqrtINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatLogINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatLog2INTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatLog10INTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatLog1pINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatExpINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatExp2INTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatExp10INTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatExpm1INTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatSinINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatCosINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

// Result value contains both values of sine and cosine and so has the size of
// 2 * Wout where Wout is equal to (1 + Eout + Mout).
template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<2 * Wout>
__spirv_ArbitraryFloatSinCosINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                  int32_t Mout, int32_t EnableSubnormals = 0,
                                  int32_t RoundingMode = 0,
                                  int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatSinPiINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatCosPiINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

// Result value contains both values of sine(A*pi) and cosine(A*pi) and so has
// the size of 2 * Wout where Wout is equal to (1 + Eout + Mout).
template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<2 * Wout>
__spirv_ArbitraryFloatSinCosPiINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                    int32_t Mout, int32_t EnableSubnormals = 0,
                                    int32_t RoundingMode = 0,
                                    int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatASinINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatASinPiINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                  int32_t Mout, int32_t EnableSubnormals = 0,
                                  int32_t RoundingMode = 0,
                                  int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatACosINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatACosPiINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                  int32_t Mout, int32_t EnableSubnormals = 0,
                                  int32_t RoundingMode = 0,
                                  int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatATanINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatATanPiINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                  int32_t Mout, int32_t EnableSubnormals = 0,
                                  int32_t RoundingMode = 0,
                                  int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatATan2INTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                 sycl::detail::ap_int<WB> B, int32_t MB,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatPowINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                               sycl::detail::ap_int<WB> B, int32_t MB,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatPowRINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                sycl::detail::ap_int<WB> B, int32_t MB,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

// PowN built-in calculates `A^B` where `A` is arbitrary precision floating
// point number and `B` is signed or unsigned arbitrary precision integer,
// i.e. its width doesn't depend on sum of exponent and mantissa.
template <int WA, int WB, int Wout>
extern __DPCPP_SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatPowNINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                sycl::detail::ap_int<WB> B, bool SignOfB,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <typename dataT>
extern __DPCPP_SYCL_EXTERNAL int32_t
__spirv_ReadPipe(__ocl_RPipeTy<dataT> Pipe, dataT *Data, int32_t Size,
                 int32_t Alignment) noexcept;
template <typename dataT>
extern __DPCPP_SYCL_EXTERNAL int32_t
__spirv_WritePipe(__ocl_WPipeTy<dataT> Pipe, const dataT *Data, int32_t Size,
                  int32_t Alignment) noexcept;
template <typename dataT>
extern __DPCPP_SYCL_EXTERNAL void
__spirv_ReadPipeBlockingINTEL(__ocl_RPipeTy<dataT> Pipe, dataT *Data,
                              int32_t Size, int32_t Alignment) noexcept;
template <typename dataT>
extern __DPCPP_SYCL_EXTERNAL void
__spirv_WritePipeBlockingINTEL(__ocl_WPipeTy<dataT> Pipe, const dataT *Data,
                               int32_t Size, int32_t Alignment) noexcept;
template <typename dataT>
extern __DPCPP_SYCL_EXTERNAL __ocl_RPipeTy<dataT>
__spirv_CreatePipeFromPipeStorage_read(
    const ConstantPipeStorage *Storage) noexcept;
template <typename dataT>
extern __DPCPP_SYCL_EXTERNAL __ocl_WPipeTy<dataT>
__spirv_CreatePipeFromPipeStorage_write(
    const ConstantPipeStorage *Storage) noexcept;

extern __DPCPP_SYCL_EXTERNAL float
__spirv_ConvertBF16ToFINTEL(uint16_t) noexcept;
extern __DPCPP_SYCL_EXTERNAL uint16_t
__spirv_ConvertFToBF16INTEL(float) noexcept;
template <int N>
extern __DPCPP_SYCL_EXTERNAL __ocl_vec_t<float, N>
    __spirv_ConvertBF16ToFINTEL(__ocl_vec_t<uint16_t, N>) noexcept;
template <int N>
extern __DPCPP_SYCL_EXTERNAL __ocl_vec_t<uint16_t, N>
    __spirv_ConvertFToBF16INTEL(__ocl_vec_t<float, N>) noexcept;

extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT void
__clc_BarrierInitialize(int64_t *state, int32_t expected_count) noexcept;

extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT void
__clc_BarrierInvalidate(int64_t *state) noexcept;

extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT int64_t
__clc_BarrierArrive(int64_t *state) noexcept;

extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT int64_t
__clc_BarrierArriveAndDrop(int64_t *state) noexcept;

extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT int64_t
__clc_BarrierArriveNoComplete(int64_t *state, int32_t count) noexcept;

extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT int64_t
__clc_BarrierArriveAndDropNoComplete(int64_t *state, int32_t count) noexcept;

extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT void
__clc_BarrierCopyAsyncArrive(int64_t *state) noexcept;

extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT void
__clc_BarrierCopyAsyncArriveNoInc(int64_t *state) noexcept;

__SYCL_CONVERGENT__ extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT void
__clc_BarrierWait(int64_t *state, int64_t arrival) noexcept;

extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT bool
__clc_BarrierTestWait(int64_t *state, int64_t arrival) noexcept;

__SYCL_CONVERGENT__ extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT void
__clc_BarrierArriveAndWait(int64_t *state) noexcept;

#if defined(__SYCL_USE_VARIADIC_SPIRV_OCL_PRINTF__) &&                         \
    !defined(__INTEL_PREVIEW_BREAKING_CHANGES)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpedantic"
#warning                                                                       \
    "__SYCL_USE_VARIADIC_SPIRV_OCL_PRINTF__ is deprecated and will be removed in a future release."
#pragma clang diagnostic pop
#endif
extern __DPCPP_SYCL_EXTERNAL int
__spirv_ocl_printf(const __attribute__((opencl_constant)) char *Format, ...);
extern __DPCPP_SYCL_EXTERNAL int __spirv_ocl_printf(const char *Format, ...);
#else
template <typename... Args>
extern __DPCPP_SYCL_EXTERNAL int
__spirv_ocl_printf(const __attribute__((opencl_constant)) char *Format,
                   Args... args);
template <typename... Args>
extern __DPCPP_SYCL_EXTERNAL int __spirv_ocl_printf(const char *Format,
                                                    Args... args);
#endif

// Native builtin extension

extern __DPCPP_SYCL_EXTERNAL float __clc_native_tanh(float);
extern __DPCPP_SYCL_EXTERNAL __ocl_vec_t<float, 2>
    __clc_native_tanh(__ocl_vec_t<float, 2>);
extern __DPCPP_SYCL_EXTERNAL __ocl_vec_t<float, 3>
    __clc_native_tanh(__ocl_vec_t<float, 3>);
extern __DPCPP_SYCL_EXTERNAL __ocl_vec_t<float, 4>
    __clc_native_tanh(__ocl_vec_t<float, 4>);
extern __DPCPP_SYCL_EXTERNAL __ocl_vec_t<float, 8>
    __clc_native_tanh(__ocl_vec_t<float, 8>);
extern __DPCPP_SYCL_EXTERNAL __ocl_vec_t<float, 16>
    __clc_native_tanh(__ocl_vec_t<float, 16>);

extern __DPCPP_SYCL_EXTERNAL _Float16 __clc_native_tanh(_Float16);
extern __DPCPP_SYCL_EXTERNAL __ocl_vec_t<_Float16, 2>
    __clc_native_tanh(__ocl_vec_t<_Float16, 2>);
extern __DPCPP_SYCL_EXTERNAL __ocl_vec_t<_Float16, 3>
    __clc_native_tanh(__ocl_vec_t<_Float16, 3>);
extern __DPCPP_SYCL_EXTERNAL __ocl_vec_t<_Float16, 4>
    __clc_native_tanh(__ocl_vec_t<_Float16, 4>);
extern __DPCPP_SYCL_EXTERNAL __ocl_vec_t<_Float16, 8>
    __clc_native_tanh(__ocl_vec_t<_Float16, 8>);
extern __DPCPP_SYCL_EXTERNAL __ocl_vec_t<_Float16, 16>
    __clc_native_tanh(__ocl_vec_t<_Float16, 16>);

extern __DPCPP_SYCL_EXTERNAL _Float16 __clc_native_exp2(_Float16);
extern __DPCPP_SYCL_EXTERNAL __ocl_vec_t<_Float16, 2>
    __clc_native_exp2(__ocl_vec_t<_Float16, 2>);
extern __DPCPP_SYCL_EXTERNAL __ocl_vec_t<_Float16, 3>
    __clc_native_exp2(__ocl_vec_t<_Float16, 3>);
extern __DPCPP_SYCL_EXTERNAL __ocl_vec_t<_Float16, 4>
    __clc_native_exp2(__ocl_vec_t<_Float16, 4>);
extern __DPCPP_SYCL_EXTERNAL __ocl_vec_t<_Float16, 8>
    __clc_native_exp2(__ocl_vec_t<_Float16, 8>);
extern __DPCPP_SYCL_EXTERNAL __ocl_vec_t<_Float16, 16>
    __clc_native_exp2(__ocl_vec_t<_Float16, 16>);

// FIXME: __clc symbols are intended to be internal symbols to libclc/libspirv
// and should not be relied upon externally; consider them deprecated. We can't,
// however, explicitly declare __spirv_ocl versions of these builtins as that
// interferes with the implicit declarations provided by clang. This results in
// legitimate calls being seen as ambiguous and causing errors. Since these
// symbols are intended to expose native versions of bfloat16 builtins for
// NVPTX, we should probably just be exposing builtins with actual bfloat16
// types, not unsigned integer types.
#define __CLC_BF16(...)                                                        \
  extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT __VA_ARGS__ __clc_fabs(           \
      __VA_ARGS__) noexcept;                                                   \
  extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT __VA_ARGS__ __clc_fmin(           \
      __VA_ARGS__, __VA_ARGS__) noexcept;                                      \
  extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT __VA_ARGS__ __clc_fmax(           \
      __VA_ARGS__, __VA_ARGS__) noexcept;                                      \
  extern __DPCPP_SYCL_EXTERNAL __SYCL_EXPORT __VA_ARGS__ __clc_fma(            \
      __VA_ARGS__, __VA_ARGS__, __VA_ARGS__) noexcept;

#define __CLC_BF16_SCAL_VEC(TYPE)                                              \
  __CLC_BF16(TYPE)                                                             \
  __CLC_BF16(__ocl_vec_t<TYPE, 2>)                                             \
  __CLC_BF16(__ocl_vec_t<TYPE, 3>)                                             \
  __CLC_BF16(__ocl_vec_t<TYPE, 4>)                                             \
  __CLC_BF16(__ocl_vec_t<TYPE, 8>)                                             \
  __CLC_BF16(__ocl_vec_t<TYPE, 16>)

__CLC_BF16_SCAL_VEC(uint16_t)
__CLC_BF16_SCAL_VEC(uint32_t)

#undef __CLC_BF16_SCAL_VEC
#undef __CLC_BF16

extern __DPCPP_SYCL_EXTERNAL int32_t __spirv_BuiltInGlobalHWThreadIDINTEL();
extern __DPCPP_SYCL_EXTERNAL int32_t __spirv_BuiltInSubDeviceIDINTEL();
extern __DPCPP_SYCL_EXTERNAL uint64_t __spirv_ReadClockKHR(int);

template <typename from, typename to>
extern __DPCPP_SYCL_EXTERNAL
    std::enable_if_t<std::is_integral_v<to> && std::is_unsigned_v<to>, to>
    __spirv_ConvertPtrToU(from val) noexcept;

template <typename T, int N>
extern __DPCPP_SYCL_EXTERNAL std::pair<__ocl_vec_t<T, N>, __ocl_vec_t<T, N>>
__spirv_IAddCarry(__ocl_vec_t<T, N> src0, __ocl_vec_t<T, N> src1);

template <typename T, int N>
extern __DPCPP_SYCL_EXTERNAL std::pair<__ocl_vec_t<T, N>, __ocl_vec_t<T, N>>
__spirv_ISubBorrow(__ocl_vec_t<T, N> src0, __ocl_vec_t<T, N> src1);
template <typename RetT, typename... ArgsT>
extern __DPCPP_SYCL_EXTERNAL __spv::__spirv_TaskSequenceINTEL *
__spirv_TaskSequenceCreateINTEL(RetT (*f)(ArgsT...), int Pipelined = -1,
                                int ClusterMode = -1,
                                unsigned int ResponseCapacity = 0,
                                unsigned int InvocationCapacity = 0) noexcept;

template <typename... ArgsT>
extern __DPCPP_SYCL_EXTERNAL void
__spirv_TaskSequenceAsyncINTEL(__spv::__spirv_TaskSequenceINTEL *TaskSequence,
                               ArgsT... Args) noexcept;

template <typename RetT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_TaskSequenceGetINTEL(
    __spv::__spirv_TaskSequenceINTEL *TaskSequence) noexcept;

extern __DPCPP_SYCL_EXTERNAL void __spirv_TaskSequenceReleaseINTEL(
    __spv::__spirv_TaskSequenceINTEL *TaskSequence) noexcept;

#endif // !__SYCL_DEVICE_ONLY__
