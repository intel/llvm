//==----------- spirv_ops.hpp --- SPIRV operations -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/__spirv/spirv_types.hpp>
#include <cstddef>
#include <cstdint>
#include <sycl/detail/defines.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/stl_type_traits.hpp>

// Convergent attribute
#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_CONVERGENT__ __attribute__((convergent))
#else
#define __SYCL_CONVERGENT__
#endif

#ifdef __SYCL_DEVICE_ONLY__

#if (SYCL_EXT_ONEAPI_MATRIX_VERSION > 1)
#define JOINT_MATRIX_INTEL(T, R, C, L, S, U)                                   \
  __spv::__spirv_JointMatrixINTEL<T, R, C, L, S, U>
#else
#define JOINT_MATRIX_INTEL(T, R, C, L, S, U)                                   \
  __spv::__spirv_JointMatrixINTEL<T, R, C, L, S>
#endif // SYCL_EXT_ONEAPI_MATRIX_VERSION

template <typename T, std::size_t R, std::size_t C,
          __spv::MatrixUse U = __spv::MatrixUse::Unnecessary,
          __spv::MatrixLayout L = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern SYCL_EXTERNAL JOINT_MATRIX_INTEL(T, R, C, L, S, U) *
__spirv_JointMatrixLoadINTEL(T *Ptr, std::size_t Stride,
                             __spv::MatrixLayout Layout = L,
                             __spv::Scope::Flag Sc = S, int MemOperand = 0);

template <typename T, std::size_t R, std::size_t C,
          __spv::MatrixUse U = __spv::MatrixUse::Unnecessary,
          __spv::MatrixLayout L = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern SYCL_EXTERNAL void __spirv_JointMatrixStoreINTEL(
    T *Ptr, JOINT_MATRIX_INTEL(T, R, C, L, S, U) *Object,
    std::size_t Stride, __spv::MatrixLayout Layout = L,
    __spv::Scope::Flag Sc = S, int MemOperand = 0);

template <typename T1, typename T2, std::size_t M, std::size_t K, std::size_t N,
          __spv::MatrixUse UA = __spv::MatrixUse::Unnecessary,
          __spv::MatrixUse UB = __spv::MatrixUse::Unnecessary,
          __spv::MatrixUse UC = __spv::MatrixUse::Unnecessary,
          __spv::MatrixLayout LA = __spv::MatrixLayout::RowMajor,
          __spv::MatrixLayout LB = __spv::MatrixLayout::RowMajor,
          __spv::MatrixLayout LC = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern SYCL_EXTERNAL JOINT_MATRIX_INTEL(T2, M, N, LC, S, UC) *
__spirv_JointMatrixMadINTEL(
    JOINT_MATRIX_INTEL(T1, M, K, LA, S, UA) *A,
    JOINT_MATRIX_INTEL(T1, K, N, LB, S, UB) *B,
    JOINT_MATRIX_INTEL(T2, M, N, LC, S, UC) *C,
    __spv::Scope::Flag Sc = __spv::Scope::Flag::Subgroup);

template <typename T1, typename T2, typename T3, std::size_t M, std::size_t K,
          std::size_t N, __spv::MatrixUse UA = __spv::MatrixUse::Unnecessary,
          __spv::MatrixUse UB = __spv::MatrixUse::Unnecessary,
          __spv::MatrixUse UC = __spv::MatrixUse::Unnecessary,
          __spv::MatrixLayout LA = __spv::MatrixLayout::RowMajor,
          __spv::MatrixLayout LB = __spv::MatrixLayout::RowMajor,
          __spv::MatrixLayout LC = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern SYCL_EXTERNAL JOINT_MATRIX_INTEL(T2, M, N, LC, S, UC) *
__spirv_JointMatrixUUMadINTEL(
    JOINT_MATRIX_INTEL(T1, M, K, LA, S, UA) *A,
    JOINT_MATRIX_INTEL(T2, K, N, LB, S, UB) *B,
    JOINT_MATRIX_INTEL(T3, M, N, LC, S, UC) *C,
    __spv::Scope::Flag Sc = __spv::Scope::Flag::Subgroup);

template <typename T1, typename T2, typename T3, std::size_t M, std::size_t K,
          std::size_t N, __spv::MatrixUse UA = __spv::MatrixUse::Unnecessary,
          __spv::MatrixUse UB = __spv::MatrixUse::Unnecessary,
          __spv::MatrixUse UC = __spv::MatrixUse::Unnecessary,
          __spv::MatrixLayout LA = __spv::MatrixLayout::RowMajor,
          __spv::MatrixLayout LB = __spv::MatrixLayout::RowMajor,
          __spv::MatrixLayout LC = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern SYCL_EXTERNAL JOINT_MATRIX_INTEL(T3, M, N, LC, S, UC) *
__spirv_JointMatrixUSMadINTEL(
    JOINT_MATRIX_INTEL(T1, M, K, LA, S, UA) *A,
    JOINT_MATRIX_INTEL(T2, K, N, LB, S, UB) *B,
    JOINT_MATRIX_INTEL(T3, M, N, LC, S, UC) *C,
    __spv::Scope::Flag Sc = __spv::Scope::Flag::Subgroup);

template <typename T1, typename T2, typename T3, std::size_t M, std::size_t K,
          std::size_t N, __spv::MatrixUse UA = __spv::MatrixUse::Unnecessary,
          __spv::MatrixUse UB = __spv::MatrixUse::Unnecessary,
          __spv::MatrixUse UC = __spv::MatrixUse::Unnecessary,
          __spv::MatrixLayout LA = __spv::MatrixLayout::RowMajor,
          __spv::MatrixLayout LB = __spv::MatrixLayout::RowMajor,
          __spv::MatrixLayout LC = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern SYCL_EXTERNAL JOINT_MATRIX_INTEL(T3, M, N, LC, S, UC) *
__spirv_JointMatrixSUMadINTEL(
    JOINT_MATRIX_INTEL(T1, M, K, LA, S, UA) *A,
    JOINT_MATRIX_INTEL(T2, K, N, LB, S, UB) *B,
    JOINT_MATRIX_INTEL(T3, M, N, LC, S, UC) *C,
    __spv::Scope::Flag Sc = __spv::Scope::Flag::Subgroup);

template <typename T, std::size_t R, std::size_t C,
          __spv::MatrixUse U = __spv::MatrixUse::Unnecessary,
          __spv::MatrixLayout L = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern SYCL_EXTERNAL JOINT_MATRIX_INTEL(T, R, C, L, S, U) *
__spirv_CompositeConstruct(const T v);

template <typename T, std::size_t R, std::size_t C,
          __spv::MatrixUse U = __spv::MatrixUse::Unnecessary,
          __spv::MatrixLayout L = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern SYCL_EXTERNAL size_t __spirv_JointMatrixWorkItemLengthINTEL(
    JOINT_MATRIX_INTEL(T, R, C, L, S, U) *);

template <typename T, std::size_t R, std::size_t C,
          __spv::MatrixUse U = __spv::MatrixUse::Unnecessary,
          __spv::MatrixLayout L = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern SYCL_EXTERNAL T __spirv_VectorExtractDynamic(
    JOINT_MATRIX_INTEL(T, R, C, L, S, U) *, size_t i);

template <typename T, std::size_t R, std::size_t C,
         __spv::MatrixUse U = __spv::MatrixUse::Unnecessary,
          __spv::MatrixLayout L = __spv::MatrixLayout::RowMajor,
          __spv::Scope::Flag S = __spv::Scope::Flag::Subgroup>
extern SYCL_EXTERNAL JOINT_MATRIX_INTEL(T, R, C, L, S, U) *
__spirv_VectorInsertDynamic(JOINT_MATRIX_INTEL(T, R, C, L, S, U) *,
                            T val, size_t i);
#undef JOINT_MATRIX_INTEL

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
  typename sycl::detail::enable_if_t<                                          \
      std::is_integral<T>::value && std::is_signed<T>::value, T>               \
      __spirv_Atomic##Op(AS T *Ptr, __spv::Scope::Flag Memory,                 \
                         __spv::MemorySemanticsMask::Flag Semantics,           \
                         T Value) {                                            \
    return __spirv_AtomicS##Op(Ptr, Memory, Semantics, Value);                 \
  }                                                                            \
  template <typename T>                                                        \
  typename sycl::detail::enable_if_t<                                          \
      std::is_integral<T>::value && !std::is_signed<T>::value, T>              \
      __spirv_Atomic##Op(AS T *Ptr, __spv::Scope::Flag Memory,                 \
                         __spv::MemorySemanticsMask::Flag Semantics,           \
                         T Value) {                                            \
    return __spirv_AtomicU##Op(Ptr, Memory, Semantics, Value);                 \
  }                                                                            \
  template <typename T>                                                        \
  typename sycl::detail::enable_if_t<std::is_floating_point<T>::value, T>      \
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
extern __attribute__((opencl_global)) dataT *
__SYCL_GenericCastToPtrExplicit_ToGlobal(void *Ptr) noexcept {
  return (__attribute__((opencl_global)) dataT *)
      __spirv_GenericCastToPtrExplicit_ToGlobal(
          Ptr, __spv::StorageClass::CrossWorkgroup);
}

template <typename dataT>
extern const __attribute__((opencl_global)) dataT *
__SYCL_GenericCastToPtrExplicit_ToGlobal(const void *Ptr) noexcept {
  return (const __attribute__((opencl_global)) dataT *)
      __spirv_GenericCastToPtrExplicit_ToGlobal(
          Ptr, __spv::StorageClass::CrossWorkgroup);
}

template <typename dataT>
extern const volatile __attribute__((opencl_global)) dataT *
__SYCL_GenericCastToPtrExplicit_ToGlobal(const volatile void *Ptr) noexcept {
  return (const volatile __attribute__((opencl_global)) dataT *)
      __spirv_GenericCastToPtrExplicit_ToGlobal(
          Ptr, __spv::StorageClass::CrossWorkgroup);
}

template <typename dataT>
extern __attribute__((opencl_local)) dataT *
__SYCL_GenericCastToPtrExplicit_ToLocal(void *Ptr) noexcept {
  return (__attribute__((opencl_local)) dataT *)
      __spirv_GenericCastToPtrExplicit_ToLocal(Ptr,
                                               __spv::StorageClass::Workgroup);
}

template <typename dataT>
extern const __attribute__((opencl_local)) dataT *
__SYCL_GenericCastToPtrExplicit_ToLocal(const void *Ptr) noexcept {
  return (const __attribute__((opencl_local)) dataT *)
      __spirv_GenericCastToPtrExplicit_ToLocal(Ptr,
                                               __spv::StorageClass::Workgroup);
}

template <typename dataT>
extern const volatile __attribute__((opencl_local)) dataT *
__SYCL_GenericCastToPtrExplicit_ToLocal(const volatile void *Ptr) noexcept {
  return (const volatile __attribute__((opencl_local)) dataT *)
      __spirv_GenericCastToPtrExplicit_ToLocal(Ptr,
                                               __spv::StorageClass::Workgroup);
}

template <typename dataT>
extern __attribute__((opencl_private)) dataT *
__SYCL_GenericCastToPtrExplicit_ToPrivate(void *Ptr) noexcept {
  return (__attribute__((opencl_private)) dataT *)
      __spirv_GenericCastToPtrExplicit_ToPrivate(Ptr,
                                                 __spv::StorageClass::Function);
}

template <typename dataT>
extern const __attribute__((opencl_private)) dataT *
__SYCL_GenericCastToPtrExplicit_ToPrivate(const void *Ptr) noexcept {
  return (const __attribute__((opencl_private)) dataT *)
      __spirv_GenericCastToPtrExplicit_ToPrivate(Ptr,
                                                 __spv::StorageClass::Function);
}

template <typename dataT>
extern const volatile __attribute__((opencl_private)) dataT *
__SYCL_GenericCastToPtrExplicit_ToPrivate(const volatile void *Ptr) noexcept {
  return (const volatile __attribute__((opencl_private)) dataT *)
      __spirv_GenericCastToPtrExplicit_ToPrivate(Ptr,
                                                 __spv::StorageClass::Function);
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
extern SYCL_EXTERNAL sycl::detail::ap_int<rW>
__spirv_FixedSqrtINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I, int32_t rI,
                       int32_t Quantization = 0, int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern SYCL_EXTERNAL sycl::detail::ap_int<rW>
__spirv_FixedRecipINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I,
                        int32_t rI, int32_t Quantization = 0,
                        int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern SYCL_EXTERNAL sycl::detail::ap_int<rW>
__spirv_FixedRsqrtINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I,
                        int32_t rI, int32_t Quantization = 0,
                        int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern SYCL_EXTERNAL sycl::detail::ap_int<rW>
__spirv_FixedSinINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I, int32_t rI,
                      int32_t Quantization = 0, int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern SYCL_EXTERNAL sycl::detail::ap_int<rW>
__spirv_FixedCosINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I, int32_t rI,
                      int32_t Quantization = 0, int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern SYCL_EXTERNAL sycl::detail::ap_int<2 * rW>
__spirv_FixedSinCosINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I,
                         int32_t rI, int32_t Quantization = 0,
                         int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern SYCL_EXTERNAL sycl::detail::ap_int<rW>
__spirv_FixedSinPiINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I,
                        int32_t rI, int32_t Quantization = 0,
                        int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern SYCL_EXTERNAL sycl::detail::ap_int<rW>
__spirv_FixedCosPiINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I,
                        int32_t rI, int32_t Quantization = 0,
                        int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern SYCL_EXTERNAL sycl::detail::ap_int<2 * rW>
__spirv_FixedSinCosPiINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I,
                           int32_t rI, int32_t Quantization = 0,
                           int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern SYCL_EXTERNAL sycl::detail::ap_int<rW>
__spirv_FixedLogINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I, int32_t rI,
                      int32_t Quantization = 0, int32_t Overflow = 0) noexcept;
template <int W, int rW>
extern SYCL_EXTERNAL sycl::detail::ap_int<rW>
__spirv_FixedExpINTEL(sycl::detail::ap_int<W> a, bool S, int32_t I, int32_t rI,
                      int32_t Quantization = 0, int32_t Overflow = 0) noexcept;

// In the following built-ins width of arbitrary precision integer type for
// a floating point variable should be equal to sum of corresponding
// exponent width E, mantissa width M and 1 for sign bit. I.e. WA = EA + MA + 1.
template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatCastINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatCastFromIntINTEL(sycl::detail::ap_int<WA> A, int32_t Mout,
                                       bool FromSign = false,
                                       int32_t EnableSubnormals = 0,
                                       int32_t RoundingMode = 0,
                                       int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatCastToIntINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                     bool ToSign = false,
                                     int32_t EnableSubnormals = 0,
                                     int32_t RoundingMode = 0,
                                     int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout> __spirv_ArbitraryFloatAddINTEL(
    sycl::detail::ap_int<WA> A, int32_t MA, sycl::detail::ap_int<WB> B,
    int32_t MB, int32_t Mout, int32_t EnableSubnormals = 0,
    int32_t RoundingMode = 0, int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout> __spirv_ArbitraryFloatSubINTEL(
    sycl::detail::ap_int<WA> A, int32_t MA, sycl::detail::ap_int<WB> B,
    int32_t MB, int32_t Mout, int32_t EnableSubnormals = 0,
    int32_t RoundingMode = 0, int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout> __spirv_ArbitraryFloatMulINTEL(
    sycl::detail::ap_int<WA> A, int32_t MA, sycl::detail::ap_int<WB> B,
    int32_t MB, int32_t Mout, int32_t EnableSubnormals = 0,
    int32_t RoundingMode = 0, int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout> __spirv_ArbitraryFloatDivINTEL(
    sycl::detail::ap_int<WA> A, int32_t MA, sycl::detail::ap_int<WB> B,
    int32_t MB, int32_t Mout, int32_t EnableSubnormals = 0,
    int32_t RoundingMode = 0, int32_t RoundingAccuracy = 0) noexcept;

// Comparison built-ins don't use Subnormal Support, Rounding Mode and
// Rounding Accuracy.
template <int WA, int WB>
extern SYCL_EXTERNAL bool
__spirv_ArbitraryFloatGTINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                              sycl::detail::ap_int<WB> B, int32_t MB) noexcept;

template <int WA, int WB>
extern SYCL_EXTERNAL bool
__spirv_ArbitraryFloatGEINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                              sycl::detail::ap_int<WB> B, int32_t MB) noexcept;

template <int WA, int WB>
extern SYCL_EXTERNAL bool
__spirv_ArbitraryFloatLTINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                              sycl::detail::ap_int<WB> B, int32_t MB) noexcept;

template <int WA, int WB>
extern SYCL_EXTERNAL bool
__spirv_ArbitraryFloatLEINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                              sycl::detail::ap_int<WB> B, int32_t MB) noexcept;

template <int WA, int WB>
extern SYCL_EXTERNAL bool
__spirv_ArbitraryFloatEQINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                              sycl::detail::ap_int<WB> B, int32_t MB) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatRecipINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatRSqrtINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatCbrtINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatHypotINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                 sycl::detail::ap_int<WB> B, int32_t MB,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatSqrtINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatLogINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatLog2INTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatLog10INTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatLog1pINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatExpINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatExp2INTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatExp10INTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatExpm1INTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatSinINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatCosINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                               int32_t Mout, int32_t EnableSubnormals = 0,
                               int32_t RoundingMode = 0,
                               int32_t RoundingAccuracy = 0) noexcept;

// Result value contains both values of sine and cosine and so has the size of
// 2 * Wout where Wout is equal to (1 + Eout + Mout).
template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<2 * Wout>
__spirv_ArbitraryFloatSinCosINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                  int32_t Mout, int32_t EnableSubnormals = 0,
                                  int32_t RoundingMode = 0,
                                  int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatSinPiINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatCosPiINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

// Result value contains both values of sine(A*pi) and cosine(A*pi) and so has
// the size of 2 * Wout where Wout is equal to (1 + Eout + Mout).
template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<2 * Wout>
__spirv_ArbitraryFloatSinCosPiINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                    int32_t Mout, int32_t EnableSubnormals = 0,
                                    int32_t RoundingMode = 0,
                                    int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatASinINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatASinPiINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                  int32_t Mout, int32_t EnableSubnormals = 0,
                                  int32_t RoundingMode = 0,
                                  int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatACosINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatACosPiINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                  int32_t Mout, int32_t EnableSubnormals = 0,
                                  int32_t RoundingMode = 0,
                                  int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatATanINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                int32_t Mout, int32_t EnableSubnormals = 0,
                                int32_t RoundingMode = 0,
                                int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatATanPiINTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                  int32_t Mout, int32_t EnableSubnormals = 0,
                                  int32_t RoundingMode = 0,
                                  int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout>
__spirv_ArbitraryFloatATan2INTEL(sycl::detail::ap_int<WA> A, int32_t MA,
                                 sycl::detail::ap_int<WB> B, int32_t MB,
                                 int32_t Mout, int32_t EnableSubnormals = 0,
                                 int32_t RoundingMode = 0,
                                 int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout> __spirv_ArbitraryFloatPowINTEL(
    sycl::detail::ap_int<WA> A, int32_t MA, sycl::detail::ap_int<WB> B,
    int32_t MB, int32_t Mout, int32_t EnableSubnormals = 0,
    int32_t RoundingMode = 0, int32_t RoundingAccuracy = 0) noexcept;

template <int WA, int WB, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout> __spirv_ArbitraryFloatPowRINTEL(
    sycl::detail::ap_int<WA> A, int32_t MA, sycl::detail::ap_int<WB> B,
    int32_t MB, int32_t Mout, int32_t EnableSubnormals = 0,
    int32_t RoundingMode = 0, int32_t RoundingAccuracy = 0) noexcept;

// PowN built-in calculates `A^B` where `A` is arbitrary precision floating
// point number and `B` is signed or unsigned arbitrary precision integer,
// i.e. its width doesn't depend on sum of exponent and mantissa.
template <int WA, int WB, int Wout>
extern SYCL_EXTERNAL sycl::detail::ap_int<Wout> __spirv_ArbitraryFloatPowNINTEL(
    sycl::detail::ap_int<WA> A, int32_t MA, sycl::detail::ap_int<WB> B,
    bool SignOfB, int32_t Mout, int32_t EnableSubnormals = 0,
    int32_t RoundingMode = 0, int32_t RoundingAccuracy = 0) noexcept;

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

extern SYCL_EXTERNAL __SYCL_EXPORT void
__clc_BarrierInitialize(int64_t *state, int32_t expected_count) noexcept;

extern SYCL_EXTERNAL __SYCL_EXPORT void
__clc_BarrierInvalidate(int64_t *state) noexcept;

extern SYCL_EXTERNAL __SYCL_EXPORT int64_t
__clc_BarrierArrive(int64_t *state) noexcept;

extern SYCL_EXTERNAL __SYCL_EXPORT int64_t
__clc_BarrierArriveAndDrop(int64_t *state) noexcept;

extern SYCL_EXTERNAL __SYCL_EXPORT int64_t
__clc_BarrierArriveNoComplete(int64_t *state, int32_t count) noexcept;

extern SYCL_EXTERNAL __SYCL_EXPORT int64_t
__clc_BarrierArriveAndDropNoComplete(int64_t *state, int32_t count) noexcept;

extern SYCL_EXTERNAL __SYCL_EXPORT void
__clc_BarrierCopyAsyncArrive(int64_t *state) noexcept;

extern SYCL_EXTERNAL __SYCL_EXPORT void
__clc_BarrierCopyAsyncArriveNoInc(int64_t *state) noexcept;

__SYCL_CONVERGENT__ extern SYCL_EXTERNAL __SYCL_EXPORT void
__clc_BarrierWait(int64_t *state, int64_t arrival) noexcept;

extern SYCL_EXTERNAL __SYCL_EXPORT bool
__clc_BarrierTestWait(int64_t *state, int64_t arrival) noexcept;

__SYCL_CONVERGENT__ extern SYCL_EXTERNAL __SYCL_EXPORT void
__clc_BarrierArriveAndWait(int64_t *state) noexcept;

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

// Native builtin extension

extern SYCL_EXTERNAL float __clc_native_tanh(float);
extern SYCL_EXTERNAL __ocl_vec_t<float, 2>
    __clc_native_tanh(__ocl_vec_t<float, 2>);
extern SYCL_EXTERNAL __ocl_vec_t<float, 3>
    __clc_native_tanh(__ocl_vec_t<float, 3>);
extern SYCL_EXTERNAL __ocl_vec_t<float, 4>
    __clc_native_tanh(__ocl_vec_t<float, 4>);
extern SYCL_EXTERNAL __ocl_vec_t<float, 8>
    __clc_native_tanh(__ocl_vec_t<float, 8>);
extern SYCL_EXTERNAL __ocl_vec_t<float, 16>
    __clc_native_tanh(__ocl_vec_t<float, 16>);

extern SYCL_EXTERNAL _Float16 __clc_native_tanh(_Float16);
extern SYCL_EXTERNAL __ocl_vec_t<_Float16, 2>
    __clc_native_tanh(__ocl_vec_t<_Float16, 2>);
extern SYCL_EXTERNAL __ocl_vec_t<_Float16, 3>
    __clc_native_tanh(__ocl_vec_t<_Float16, 3>);
extern SYCL_EXTERNAL __ocl_vec_t<_Float16, 4>
    __clc_native_tanh(__ocl_vec_t<_Float16, 4>);
extern SYCL_EXTERNAL __ocl_vec_t<_Float16, 8>
    __clc_native_tanh(__ocl_vec_t<_Float16, 8>);
extern SYCL_EXTERNAL __ocl_vec_t<_Float16, 16>
    __clc_native_tanh(__ocl_vec_t<_Float16, 16>);

extern SYCL_EXTERNAL _Float16 __clc_native_exp2(_Float16);
extern SYCL_EXTERNAL __ocl_vec_t<_Float16, 2>
    __clc_native_exp2(__ocl_vec_t<_Float16, 2>);
extern SYCL_EXTERNAL __ocl_vec_t<_Float16, 3>
    __clc_native_exp2(__ocl_vec_t<_Float16, 3>);
extern SYCL_EXTERNAL __ocl_vec_t<_Float16, 4>
    __clc_native_exp2(__ocl_vec_t<_Float16, 4>);
extern SYCL_EXTERNAL __ocl_vec_t<_Float16, 8>
    __clc_native_exp2(__ocl_vec_t<_Float16, 8>);
extern SYCL_EXTERNAL __ocl_vec_t<_Float16, 16>
    __clc_native_exp2(__ocl_vec_t<_Float16, 16>);

#define __CLC_BF16(...)                                                        \
  extern SYCL_EXTERNAL __SYCL_EXPORT __VA_ARGS__ __clc_fabs(                   \
      __VA_ARGS__) noexcept;                                                   \
  extern SYCL_EXTERNAL __SYCL_EXPORT __VA_ARGS__ __clc_fmin(                   \
      __VA_ARGS__, __VA_ARGS__) noexcept;                                      \
  extern SYCL_EXTERNAL __SYCL_EXPORT __VA_ARGS__ __clc_fmax(                   \
      __VA_ARGS__, __VA_ARGS__) noexcept;                                      \
  extern SYCL_EXTERNAL __SYCL_EXPORT __VA_ARGS__ __clc_fma(                    \
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
