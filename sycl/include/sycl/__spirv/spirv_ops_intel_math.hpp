//==---- spirv_ops_intel_math.hpp --- SPIRV INTEL numeric operations ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_ops_builtin_decls.hpp>

#include <cstdint>

#ifdef __SYCL_DEVICE_ONLY__

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

#endif