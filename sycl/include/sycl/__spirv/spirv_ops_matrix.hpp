//==------- spirv_ops_matrix.hpp --- SPIRV matrix operations --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_ops_base.hpp>

#include <cstddef>
#include <cstdint>

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

#endif