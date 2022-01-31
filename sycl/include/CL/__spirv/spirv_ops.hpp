//==----------- spirv_ops.hpp --- SPIRV operations -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/__spirv/spirv_types.hpp>
#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/stl_type_traits.hpp>
#include <cstddef>
#include <cstdint>

// Convergent attribute
#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_CONVERGENT__ __attribute__((convergent))
#else
#define __SYCL_CONVERGENT__
#endif

#ifdef __SYCL_DEVICE_ONLY__
template <typename T, std::size_t R, std::size_t C,
          __spv::MatrixLayout L = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern SYCL_EXTERNAL __spv::__spirv_JointMatrixINTEL<T, R, C, L, S> *
__spirv_JointMatrixLoadINTEL(T *Ptr, std::size_t Stride,
                             __spv::MatrixLayout Layout = L,
                             __spv::Scope::Flag Sc = S, int MemOperand = 0);

template <typename T, std::size_t R, std::size_t C,
          __spv::MatrixLayout L = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern SYCL_EXTERNAL void __spirv_JointMatrixStoreINTEL(
    T *Ptr, __spv::__spirv_JointMatrixINTEL<T, R, C, L, S> *Object,
    std::size_t Stride, __spv::MatrixLayout Layout = L,
    __spv::Scope::Flag Sc = S, int MemOperand = 0);

template <typename T1, typename T2, std::size_t M, std::size_t K, std::size_t N,
          __spv::MatrixLayout LA = __spv::MatrixLayout::RowMajor,
          __spv::MatrixLayout LB = __spv::MatrixLayout::RowMajor,
          __spv::MatrixLayout LC = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern SYCL_EXTERNAL __spv::__spirv_JointMatrixINTEL<T2, M, N, LC, S> *
__spirv_JointMatrixMadINTEL(
    __spv::__spirv_JointMatrixINTEL<T1, M, K, LA, S> *A,
    __spv::__spirv_JointMatrixINTEL<T1, K, N, LB, S> *B,
    __spv::__spirv_JointMatrixINTEL<T2, M, N, LC, S> *C,
    __spv::Scope::Flag Sc = __spv::Scope::Flag::Subgroup);

template <typename T1, typename T2, typename T3, std::size_t M, std::size_t K,
          std::size_t N, __spv::MatrixLayout LA = __spv::MatrixLayout::RowMajor,
          __spv::MatrixLayout LB = __spv::MatrixLayout::RowMajor,
          __spv::MatrixLayout LC = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern SYCL_EXTERNAL __spv::__spirv_JointMatrixINTEL<T3, M, N, LC, S> *
__spirv_JointMatrixUUMadINTEL(
    __spv::__spirv_JointMatrixINTEL<T1, M, K, LA, S> *A,
    __spv::__spirv_JointMatrixINTEL<T2, K, N, LB, S> *B,
    __spv::__spirv_JointMatrixINTEL<T3, M, N, LC, S> *C,
    __spv::Scope::Flag Sc = __spv::Scope::Flag::Subgroup);

template <typename T1, typename T2, typename T3, std::size_t M, std::size_t K,
          std::size_t N, __spv::MatrixLayout LA = __spv::MatrixLayout::RowMajor,
          __spv::MatrixLayout LB = __spv::MatrixLayout::RowMajor,
          __spv::MatrixLayout LC = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern SYCL_EXTERNAL __spv::__spirv_JointMatrixINTEL<T3, M, N, LC, S> *
__spirv_JointMatrixUSMadINTEL(
    __spv::__spirv_JointMatrixINTEL<T1, M, K, LA, S> *A,
    __spv::__spirv_JointMatrixINTEL<T2, K, N, LB, S> *B,
    __spv::__spirv_JointMatrixINTEL<T3, M, N, LC, S> *C,
    __spv::Scope::Flag Sc = __spv::Scope::Flag::Subgroup);

template <typename T1, typename T2, typename T3, std::size_t M, std::size_t K,
          std::size_t N, __spv::MatrixLayout LA = __spv::MatrixLayout::RowMajor,
          __spv::MatrixLayout LB = __spv::MatrixLayout::RowMajor,
          __spv::MatrixLayout LC = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern SYCL_EXTERNAL __spv::__spirv_JointMatrixINTEL<T3, M, N, LC, S> *
__spirv_JointMatrixSUMadINTEL(
    __spv::__spirv_JointMatrixINTEL<T1, M, K, LA, S> *A,
    __spv::__spirv_JointMatrixINTEL<T2, K, N, LB, S> *B,
    __spv::__spirv_JointMatrixINTEL<T3, M, N, LC, S> *C,
    __spv::Scope::Flag Sc = __spv::Scope::Flag::Subgroup);

template <typename T, std::size_t R, std::size_t C,
          __spv::MatrixLayout L = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern SYCL_EXTERNAL __spv::__spirv_JointMatrixINTEL<T, R, C, L, S> *
__spirv_CompositeConstruct(const T v);

template <typename T, std::size_t R, std::size_t C, __spv::MatrixLayout U,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern SYCL_EXTERNAL size_t __spirv_JointMatrixWorkItemLengthINTEL(
    __spv::__spirv_JointMatrixINTEL<T, R, C, U, S> *);

template <typename T, std::size_t R, std::size_t C, __spv::MatrixLayout U,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern SYCL_EXTERNAL T __spirv_VectorExtractDynamic(
    __spv::__spirv_JointMatrixINTEL<T, R, C, U, S> *, size_t i);

template <typename T, std::size_t R, std::size_t C, __spv::MatrixLayout U,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern SYCL_EXTERNAL __spv::__spirv_JointMatrixINTEL<T, R, C, U, S> *
__spirv_VectorInsertDynamic(__spv::__spirv_JointMatrixINTEL<T, R, C, U, S> *,
                            T val, size_t i);

#ifndef __SPIRV_BUILTIN_DECLARATIONS__
#error                                                                         \
    "SPIR-V built-ins are not available. Please set -fdeclare-spirv-builtins flag."
#endif

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

#define __SYCL_OpGroupAsyncCopyGlobalToLocal __spirv_GroupAsyncCopy
#define __SYCL_OpGroupAsyncCopyLocalToGlobal __spirv_GroupAsyncCopy

// Atomic SPIR-V builtins
#define __SPIRV_ATOMIC_LOAD(AS, Type)                                          \
  extern SYCL_EXTERNAL Type __spirv_AtomicLoad(                                \
      AS const Type *P, __spv::Scope::Flag S,                                  \
      __spv::MemorySemanticsMask::Flag O);
#define __SPIRV_ATOMIC_STORE(AS, Type)                                         \
  extern SYCL_EXTERNAL void __spirv_AtomicStore(                               \
      AS Type *P, __spv::Scope::Flag S, __spv::MemorySemanticsMask::Flag O,    \
      Type V);
#define __SPIRV_ATOMIC_EXCHANGE(AS, Type)                                      \
  extern SYCL_EXTERNAL Type __spirv_AtomicExchange(                            \
      AS Type *P, __spv::Scope::Flag S, __spv::MemorySemanticsMask::Flag O,    \
      Type V);
#define __SPIRV_ATOMIC_CMP_EXCHANGE(AS, Type)                                  \
  extern SYCL_EXTERNAL Type __spirv_AtomicCompareExchange(                     \
      AS Type *P, __spv::Scope::Flag S, __spv::MemorySemanticsMask::Flag E,    \
      __spv::MemorySemanticsMask::Flag U, Type V, Type C);
#define __SPIRV_ATOMIC_IADD(AS, Type)                                          \
  extern SYCL_EXTERNAL Type __spirv_AtomicIAdd(                                \
      AS Type *P, __spv::Scope::Flag S, __spv::MemorySemanticsMask::Flag O,    \
      Type V);
#define __SPIRV_ATOMIC_ISUB(AS, Type)                                          \
  extern SYCL_EXTERNAL Type __spirv_AtomicISub(                                \
      AS Type *P, __spv::Scope::Flag S, __spv::MemorySemanticsMask::Flag O,    \
      Type V);
#define __SPIRV_ATOMIC_FADD(AS, Type)                                          \
  extern SYCL_EXTERNAL Type __spirv_AtomicFAddEXT(                             \
      AS Type *P, __spv::Scope::Flag S, __spv::MemorySemanticsMask::Flag O,    \
      Type V);
#define __SPIRV_ATOMIC_SMIN(AS, Type)                                          \
  extern SYCL_EXTERNAL Type __spirv_AtomicSMin(                                \
      AS Type *P, __spv::Scope::Flag S, __spv::MemorySemanticsMask::Flag O,    \
      Type V);
#define __SPIRV_ATOMIC_UMIN(AS, Type)                                          \
  extern SYCL_EXTERNAL Type __spirv_AtomicUMin(                                \
      AS Type *P, __spv::Scope::Flag S, __spv::MemorySemanticsMask::Flag O,    \
      Type V);
#define __SPIRV_ATOMIC_FMIN(AS, Type)                                          \
  extern SYCL_EXTERNAL Type __spirv_AtomicFMinEXT(                             \
      AS Type *P, __spv::Scope::Flag S, __spv::MemorySemanticsMask::Flag O,    \
      Type V);
#define __SPIRV_ATOMIC_SMAX(AS, Type)                                          \
  extern SYCL_EXTERNAL Type __spirv_AtomicSMax(                                \
      AS Type *P, __spv::Scope::Flag S, __spv::MemorySemanticsMask::Flag O,    \
      Type V);
#define __SPIRV_ATOMIC_UMAX(AS, Type)                                          \
  extern SYCL_EXTERNAL Type __spirv_AtomicUMax(                                \
      AS Type *P, __spv::Scope::Flag S, __spv::MemorySemanticsMask::Flag O,    \
      Type V);
#define __SPIRV_ATOMIC_FMAX(AS, Type)                                          \
  extern SYCL_EXTERNAL Type __spirv_AtomicFMaxEXT(                             \
      AS Type *P, __spv::Scope::Flag S, __spv::MemorySemanticsMask::Flag O,    \
      Type V);
#define __SPIRV_ATOMIC_AND(AS, Type)                                           \
  extern SYCL_EXTERNAL Type __spirv_AtomicAnd(                                 \
      AS Type *P, __spv::Scope::Flag S, __spv::MemorySemanticsMask::Flag O,    \
      Type V);
#define __SPIRV_ATOMIC_OR(AS, Type)                                            \
  extern SYCL_EXTERNAL Type __spirv_AtomicOr(                                  \
      AS Type *P, __spv::Scope::Flag S, __spv::MemorySemanticsMask::Flag O,    \
      Type V);
#define __SPIRV_ATOMIC_XOR(AS, Type)                                           \
  extern SYCL_EXTERNAL Type __spirv_AtomicXor(                                 \
      AS Type *P, __spv::Scope::Flag S, __spv::MemorySemanticsMask::Flag O,    \
      Type V);

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
  typename cl::sycl::detail::enable_if_t<                                      \
      std::is_integral<T>::value && std::is_signed<T>::value, T>               \
      __spirv_Atomic##Op(AS T *Ptr, __spv::Scope::Flag Memory,                 \
                         __spv::MemorySemanticsMask::Flag Semantics,           \
                         T Value) {                                            \
    return __spirv_AtomicS##Op(Ptr, Memory, Semantics, Value);                 \
  }                                                                            \
  template <typename T>                                                        \
  typename cl::sycl::detail::enable_if_t<                                      \
      std::is_integral<T>::value && !std::is_signed<T>::value, T>              \
      __spirv_Atomic##Op(AS T *Ptr, __spv::Scope::Flag Memory,                 \
                         __spv::MemorySemanticsMask::Flag Semantics,           \
                         T Value) {                                            \
    return __spirv_AtomicU##Op(Ptr, Memory, Semantics, Value);                 \
  }                                                                            \
  template <typename T>                                                        \
  typename cl::sycl::detail::enable_if_t<std::is_floating_point<T>::value, T>  \
      __spirv_Atomic##Op(AS T *Ptr, __spv::Scope::Flag Memory,                 \
                         __spv::MemorySemanticsMask::Flag Semantics,           \
                         T Value) {                                            \
    return __spirv_AtomicF##Op##EXT(Ptr, Memory, Semantics, Value);            \
  }

#define __SPIRV_ATOMICS(macro, Arg)                                            \
  macro(__attribute__((opencl_global)), Arg)                                   \
      macro(__attribute__((opencl_local)), Arg) macro(, Arg)

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

extern SYCL_EXTERNAL __attribute__((opencl_global)) void *
__spirv_GenericCastToPtrExplicit_ToGlobal(const void *Ptr,
                                          __spv::StorageClass::Flag S) noexcept;

extern SYCL_EXTERNAL __attribute__((opencl_local)) void *
__spirv_GenericCastToPtrExplicit_ToLocal(const void *Ptr,
                                         __spv::StorageClass::Flag S) noexcept;

template <typename dataT>
extern __attribute__((opencl_global)) dataT *
__SYCL_GenericCastToPtrExplicit_ToGlobal(const void *Ptr) noexcept {
  return (__attribute__((opencl_global)) dataT *)
      __spirv_GenericCastToPtrExplicit_ToGlobal(
          Ptr, __spv::StorageClass::CrossWorkgroup);
}

template <typename dataT>
extern __attribute__((opencl_local)) dataT *
__SYCL_GenericCastToPtrExplicit_ToLocal(const void *Ptr) noexcept {
  return (__attribute__((opencl_local)) dataT *)
      __spirv_GenericCastToPtrExplicit_ToLocal(Ptr,
                                               __spv::StorageClass::Workgroup);
}

template <typename dataT>
__SYCL_CONVERGENT__ extern SYCL_EXTERNAL dataT
__spirv_SubgroupShuffleINTEL(dataT Data, uint32_t InvocationId) noexcept;
template <typename dataT>
__SYCL_CONVERGENT__ extern SYCL_EXTERNAL dataT __spirv_SubgroupShuffleDownINTEL(
    dataT Current, dataT Next, uint32_t Delta) noexcept;
template <typename dataT>
__SYCL_CONVERGENT__ extern SYCL_EXTERNAL dataT __spirv_SubgroupShuffleUpINTEL(
    dataT Previous, dataT Current, uint32_t Delta) noexcept;
template <typename dataT>
__SYCL_CONVERGENT__ extern SYCL_EXTERNAL dataT
__spirv_SubgroupShuffleXorINTEL(dataT Data, uint32_t Value) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern SYCL_EXTERNAL dataT __spirv_SubgroupBlockReadINTEL(
    const __attribute__((opencl_global)) uint8_t *Ptr) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern SYCL_EXTERNAL void
__spirv_SubgroupBlockWriteINTEL(__attribute__((opencl_global)) uint8_t *Ptr,
                                dataT Data) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern SYCL_EXTERNAL dataT __spirv_SubgroupBlockReadINTEL(
    const __attribute__((opencl_global)) uint16_t *Ptr) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern SYCL_EXTERNAL void
__spirv_SubgroupBlockWriteINTEL(__attribute__((opencl_global)) uint16_t *Ptr,
                                dataT Data) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern SYCL_EXTERNAL dataT __spirv_SubgroupBlockReadINTEL(
    const __attribute__((opencl_global)) uint32_t *Ptr) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern SYCL_EXTERNAL void
__spirv_SubgroupBlockWriteINTEL(__attribute__((opencl_global)) uint32_t *Ptr,
                                dataT Data) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern SYCL_EXTERNAL dataT __spirv_SubgroupBlockReadINTEL(
    const __attribute__((opencl_global)) uint64_t *Ptr) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern SYCL_EXTERNAL void
__spirv_SubgroupBlockWriteINTEL(__attribute__((opencl_global)) uint64_t *Ptr,
                                dataT Data) noexcept;
template <int W, int rW>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<rW>
__spirv_FixedSqrtINTEL(cl::sycl::detail::ap_int<W> a, bool S, int32_t I,
                       int32_t rI, int32_t Quantization = 0,
                       int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<rW>
__spirv_FixedRecipINTEL(cl::sycl::detail::ap_int<W> a, bool S, int32_t I,
                        int32_t rI, int32_t Quantization = 0,
                        int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<rW>
__spirv_FixedRsqrtINTEL(cl::sycl::detail::ap_int<W> a, bool S, int32_t I,
                        int32_t rI, int32_t Quantization = 0,
                        int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<rW>
__spirv_FixedSinINTEL(cl::sycl::detail::ap_int<W> a, bool S, int32_t I,
                      int32_t rI, int32_t Quantization = 0,
                      int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<rW>
__spirv_FixedCosINTEL(cl::sycl::detail::ap_int<W> a, bool S, int32_t I,
                      int32_t rI, int32_t Quantization = 0,
                      int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<2 * rW>
__spirv_FixedSinCosINTEL(cl::sycl::detail::ap_int<W> a, bool S, int32_t I,
                         int32_t rI, int32_t Quantization = 0,
                         int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<rW>
__spirv_FixedSinPiINTEL(cl::sycl::detail::ap_int<W> a, bool S, int32_t I,
                        int32_t rI, int32_t Quantization = 0,
                        int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<rW>
__spirv_FixedCosPiINTEL(cl::sycl::detail::ap_int<W> a, bool S, int32_t I,
                        int32_t rI, int32_t Quantization = 0,
                        int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<2 * rW>
__spirv_FixedSinCosPiINTEL(cl::sycl::detail::ap_int<W> a, bool S, int32_t I,
                           int32_t rI, int32_t Quantization = 0,
                           int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<rW>
__spirv_FixedLogINTEL(cl::sycl::detail::ap_int<W> a, bool S, int32_t I,
                      int32_t rI, int32_t Quantization = 0,
                      int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<rW>
__spirv_FixedExpINTEL(cl::sycl::detail::ap_int<W> a, bool S, int32_t I,
                      int32_t rI, int32_t Quantization = 0,
                      int32_t Overflow = 0) noexcept;

// In the following built-ins width of arbitrary precision integer type for
// a floating point variable should be equal to sum of corresponding
// exponent width E, mantissa width M and 1 for sign bit. I.e. WA = EA + MA + 1.
template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatCastINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatCastFromIntINTEL(cl::sycl::detail::ap_int<WA> A,
                                       int32_t Mout, bool FromSign = false,
                                       int32_t EnableSubnormals = 0,
                                       int32_t RoundingMode = 0,
                                       int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatCastToIntINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                     bool ToSign = false,
                                     int32_t EnableSubnormals = 0,
                                     int32_t RoundingMode = 0,
                                     int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatAddINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                               cl::sycl::detail::ap_int<WB> B, int32_t MB,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatSubINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                               cl::sycl::detail::ap_int<WB> B, int32_t MB,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatMulINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                               cl::sycl::detail::ap_int<WB> B, int32_t MB,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatDivINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                               cl::sycl::detail::ap_int<WB> B, int32_t MB,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

// Comparison built-ins don't use Subnormal Support, Rounding Mode and
// Rounding Accuracy.
template <int WA, int WB>
extern SYCL_EXTERNAL bool
__spirv_ArbitraryFloatGTINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                              cl::sycl::detail::ap_int<WB> B,
                              int32_t MB) noexcept;

template <int WA, int WB>
extern SYCL_EXTERNAL bool
__spirv_ArbitraryFloatGEINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                              cl::sycl::detail::ap_int<WB> B,
                              int32_t MB) noexcept;

template <int WA, int WB>
extern SYCL_EXTERNAL bool
__spirv_ArbitraryFloatLTINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                              cl::sycl::detail::ap_int<WB> B,
                              int32_t MB) noexcept;

template <int WA, int WB>
extern SYCL_EXTERNAL bool
__spirv_ArbitraryFloatLEINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                              cl::sycl::detail::ap_int<WB> B,
                              int32_t MB) noexcept;

template <int WA, int WB>
extern SYCL_EXTERNAL bool
__spirv_ArbitraryFloatEQINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                              cl::sycl::detail::ap_int<WB> B,
                              int32_t MB) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatRecipINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatRSqrtINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatCbrtINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatHypotINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                 cl::sycl::detail::ap_int<WB> B, int32_t MB,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatSqrtINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatLogINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatLog2INTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatLog10INTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatLog1pINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatExpINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatExp2INTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatExp10INTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatExpm1INTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatSinINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatCosINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

// Result value contains both values of sine and cosine and so has the size of
// 2 * Wout where Wout is equal to (1 + Eout + Mout).
template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<2 * Wout>
__spirv_ArbitraryFloatSinCosINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                  int32_t Mout, int32_t EnableSubnormals = 0,
                                  int32_t RoundingMode = 0,
                                  int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatSinPiINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatCosPiINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

// Result value contains both values of sine(A*pi) and cosine(A*pi) and so has
// the size of 2 * Wout where Wout is equal to (1 + Eout + Mout).
template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<2 * Wout>
__spirv_ArbitraryFloatSinCosPiINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                    int32_t Mout, int32_t EnableSubnormals = 0,
                                    int32_t RoundingMode = 0,
                                    int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatASinINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatASinPiINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                  int32_t Mout, int32_t EnableSubnormals = 0,
                                  int32_t RoundingMode = 0,
                                  int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatACosINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatACosPiINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                  int32_t Mout, int32_t EnableSubnormals = 0,
                                  int32_t RoundingMode = 0,
                                  int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatATanINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatATanPiINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                  int32_t Mout, int32_t EnableSubnormals = 0,
                                  int32_t RoundingMode = 0,
                                  int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatATan2INTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                 cl::sycl::detail::ap_int<WB> B, int32_t MB,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatPowINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                               cl::sycl::detail::ap_int<WB> B, int32_t MB,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatPowRINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                cl::sycl::detail::ap_int<WB> B, int32_t MB,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

// PowN built-in calculates `A^B` where `A` is arbitrary precision floating
// point number and `B` is signed or unsigned arbitrary precision integer,
// i.e. its width doesn't depend on sum of exponent and mantissa.
template <int WA, int WB, int Wout>
extern SYCL_EXTERNAL cl::sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatPowNINTEL(cl::sycl::detail::ap_int<WA> A, int32_t MA,
                                cl::sycl::detail::ap_int<WB> B, bool SignOfB,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <typename dataT>
extern SYCL_EXTERNAL int32_t __spirv_ReadPipe(__ocl_RPipeTy<dataT> Pipe,
                                              dataT *Data, int32_t Size,
                                              int32_t Alignment) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL int32_t __spirv_WritePipe(__ocl_WPipeTy<dataT> Pipe,
                                               const dataT *Data, int32_t Size,
                                               int32_t Alignment) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL void
__spirv_ReadPipeBlockingINTEL(__ocl_RPipeTy<dataT> Pipe, dataT *Data,
                              int32_t Size, int32_t Alignment) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL void
__spirv_WritePipeBlockingINTEL(__ocl_WPipeTy<dataT> Pipe, const dataT *Data,
                               int32_t Size, int32_t Alignment) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL __ocl_RPipeTy<dataT>
__spirv_CreatePipeFromPipeStorage_read(
    const ConstantPipeStorage *Storage) noexcept;
template <typename dataT>
extern SYCL_EXTERNAL __ocl_WPipeTy<dataT>
__spirv_CreatePipeFromPipeStorage_write(
    const ConstantPipeStorage *Storage) noexcept;

extern SYCL_EXTERNAL void
__spirv_ocl_prefetch(const __attribute__((opencl_global)) char *Ptr,
                     size_t NumBytes) noexcept;

extern SYCL_EXTERNAL uint16_t __spirv_ConvertFToBF16INTEL(float) noexcept;
extern SYCL_EXTERNAL float __spirv_ConvertBF16ToFINTEL(uint16_t) noexcept;

__SYCL_CONVERGENT__ extern SYCL_EXTERNAL __SYCL_EXPORT __ocl_vec_t<uint32_t, 4>
__spirv_GroupNonUniformBallot(uint32_t Execution, bool Predicate) noexcept;

#ifdef __SYCL_USE_NON_VARIADIC_SPIRV_OCL_PRINTF__
template <typename... Args>
extern SYCL_EXTERNAL int
__spirv_ocl_printf(const __attribute__((opencl_constant)) char *Format,
                   Args... args);
template <typename... Args>
extern SYCL_EXTERNAL int __spirv_ocl_printf(const char *Format, Args... args);
#else
extern SYCL_EXTERNAL int
__spirv_ocl_printf(const __attribute__((opencl_constant)) char *Format, ...);
extern SYCL_EXTERNAL int __spirv_ocl_printf(const char *Format, ...);
#endif

#else // if !__SYCL_DEVICE_ONLY__

template <typename dataT>
__SYCL_CONVERGENT__ extern __ocl_event_t
__SYCL_OpGroupAsyncCopyGlobalToLocal(__spv::Scope::Flag, dataT *Dest,
                                     dataT *Src, size_t NumElements,
                                     size_t Stride, __ocl_event_t) noexcept {
  for (size_t i = 0; i < NumElements; i++) {
    Dest[i] = Src[i * Stride];
  }
  // A real instance of the class is not needed, return dummy pointer.
  return nullptr;
}

template <typename dataT>
__SYCL_CONVERGENT__ extern __ocl_event_t
__SYCL_OpGroupAsyncCopyLocalToGlobal(__spv::Scope::Flag, dataT *Dest,
                                     dataT *Src, size_t NumElements,
                                     size_t Stride, __ocl_event_t) noexcept {
  for (size_t i = 0; i < NumElements; i++) {
    Dest[i * Stride] = Src[i];
  }
  // A real instance of the class is not needed, return dummy pointer.
  return nullptr;
}

extern __SYCL_EXPORT void __spirv_ocl_prefetch(const char *Ptr,
                                               size_t NumBytes) noexcept;

__SYCL_CONVERGENT__ extern SYCL_EXTERNAL __SYCL_EXPORT void
__spirv_ControlBarrier(__spv::Scope Execution, __spv::Scope Memory,
                       uint32_t Semantics) noexcept;

__SYCL_CONVERGENT__ extern SYCL_EXTERNAL __SYCL_EXPORT void
__spirv_MemoryBarrier(__spv::Scope Memory, uint32_t Semantics) noexcept;

__SYCL_CONVERGENT__ extern SYCL_EXTERNAL __SYCL_EXPORT void
__spirv_GroupWaitEvents(__spv::Scope Execution, uint32_t NumEvents,
                        __ocl_event_t *WaitEvents) noexcept;

#endif // !__SYCL_DEVICE_ONLY__
