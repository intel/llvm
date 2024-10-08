//==- fpga_ihs_float.cpp - SYCL FPGA arbitrary precision floating point test
//-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -I %sycl_include -S -emit-llvm -fsycl -fsycl-device-only -Xclang -no-enable-noundef-analysis %s -o - | FileCheck %s
// RUN: %clangxx -I %sycl_include -S -emit-llvm -fsycl -fno-sycl-early-optimizations -fsycl-device-only -Xclang -no-enable-noundef-analysis %s -o - | FileCheck %s

#include <sycl/sycl.hpp>

constexpr int32_t Subnorm = 0;
constexpr int32_t RndMode = 2;
constexpr int32_t RndAcc = 1;
constexpr bool FromSign = false;
constexpr bool ToSign = true;
constexpr bool SignOfB = false;

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_cast(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatCastINTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
}
// CHECK: call spir_func i40 @_Z{{[0-9]+}}__spirv_ArbitraryFloatCastINTEL{{.*}}(i40 {{[%A-Za-z0-9.]+}}, i32 28, i32 30, i32 0, i32 2, i32 1)
template auto ap_float_cast<11, 28, 9, 30>(sycl::detail::ap_int<1 + 11 + 28> A);

template <int WA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_cast_from_int(sycl::detail::ap_int<WA> A) {
  return __spirv_ArbitraryFloatCastFromIntINTEL<WA, 1 + Eout + Mout>(
      A, Mout, FromSign, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i25 @_Z{{[0-9]+}}__spirv_ArbitraryFloatCastFromIntINTEL{{.*}}(i43 {{[%A-Za-z0-9.]+}}, i32 16, i1 zeroext false, i32 0, i32 2, i32 1)
}
template auto ap_float_cast_from_int<43, 8, 16>(sycl::detail::ap_int<43> A);

template <int EA, int MA, int Wout>
SYCL_EXTERNAL auto ap_float_cast_to_int(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatCastToIntINTEL<1 + EA + MA, Wout>(
      A, MA, ToSign, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i30 @_Z{{[0-9]+}}__spirv_ArbitraryFloatCastToIntINTEL{{.*}}(i23 signext {{[%A-Za-z0-9.]+}}, i32 15, i1 zeroext true, i32 0, i32 2, i32 1)
}
template auto ap_float_cast_to_int<7, 15, 30>(sycl::detail::ap_int<1 + 7 + 15>);

template <int EA, int MA, int EB, int MB, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_add(sycl::detail::ap_int<1 + EA + MA> A,
                                sycl::detail::ap_int<1 + EB + MB> B) {
  return __spirv_ArbitraryFloatAddINTEL<1 + EA + MA, 1 + EB + MB,
                                        1 + Eout + Mout>(
      A, MA, B, MB, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i14 @_Z{{[0-9]+}}__spirv_ArbitraryFloatAddINTEL{{.*}}(i13 signext {{[%A-Za-z0-9.]+}}, i32 7, i15 signext {{[%A-Za-z0-9.]+}}, i32 8, i32 9, i32 0, i32 2, i32 1)
  // CHECK: call spir_func signext i13 @_Z{{[0-9]+}}__spirv_ArbitraryFloatAddINTEL{{.*}}(i15 signext {{[%A-Za-z0-9.]+}}, i32 8, i14 signext {{[%A-Za-z0-9.]+}}, i32 9, i32 7, i32 0, i32 2, i32 1)
}
template auto ap_float_add<5, 7, 6, 8, 4, 9>(sycl::detail::ap_int<1 + 5 + 7> A,
                                             sycl::detail::ap_int<1 + 6 + 8> B);
template auto ap_float_add<6, 8, 4, 9, 5, 7>(sycl::detail::ap_int<1 + 6 + 8> A,
                                             sycl::detail::ap_int<1 + 4 + 9> B);

template <int EA, int MA, int EB, int MB, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_sub(sycl::detail::ap_int<1 + EA + MA> A,
                                sycl::detail::ap_int<1 + EB + MB> B) {
  return __spirv_ArbitraryFloatSubINTEL<1 + EA + MA, 1 + EB + MB,
                                        1 + Eout + Mout>(
      A, MA, B, MB, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i13 @_Z{{[0-9]+}}__spirv_ArbitraryFloatSubINTEL{{.*}}(i9 signext {{[%A-Za-z0-9.]+}}, i32 4, i11 signext {{[%A-Za-z0-9.]+}}, i32 5, i32 6, i32 0, i32 2, i32 1)
}
template auto ap_float_sub<4, 4, 5, 5, 6, 6>(sycl::detail::ap_int<1 + 4 + 4> A,
                                             sycl::detail::ap_int<1 + 5 + 5> B);

template <int EA, int MA, int EB, int MB, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_mul(sycl::detail::ap_int<1 + EA + MA> A,
                                sycl::detail::ap_int<1 + EB + MB> B) {
  return __spirv_ArbitraryFloatMulINTEL<1 + EA + MA, 1 + EB + MB,
                                        1 + Eout + Mout>(
      A, MA, B, MB, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i51 @_Z{{[0-9]+}}__spirv_ArbitraryFloatMulINTEL{{.*}}(i51 {{[%A-Za-z0-9.]+}}, i32 34, i51 {{[%A-Za-z0-9.]+}}, i32 34, i32 34, i32 0, i32 2, i32 1)
}
template auto
ap_float_mul<16, 34, 16, 34, 16, 34>(sycl::detail::ap_int<1 + 16 + 34> A,
                                     sycl::detail::ap_int<1 + 16 + 34> B);

template <int EA, int MA, int EB, int MB, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_div(sycl::detail::ap_int<1 + EA + MA> A,
                                sycl::detail::ap_int<1 + EB + MB> B) {
  return __spirv_ArbitraryFloatDivINTEL<1 + EA + MA, 1 + EB + MB,
                                        1 + Eout + Mout>(
      A, MA, B, MB, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i18 @_Z{{[0-9]+}}__spirv_ArbitraryFloatDivINTEL{{.*}}(i16 signext {{[%A-Za-z0-9.]+}}, i32 11, i16 signext {{[%A-Za-z0-9.]+}}, i32 11, i32 12, i32 0, i32 2, i32 1)
}
template auto
ap_float_div<4, 11, 4, 11, 5, 12>(sycl::detail::ap_int<1 + 4 + 11> A,
                                  sycl::detail::ap_int<1 + 4 + 11> B);

template <int EA, int MA, int EB, int MB>
SYCL_EXTERNAL auto ap_float_gt(sycl::detail::ap_int<1 + EA + MA> A,
                               sycl::detail::ap_int<1 + EB + MB> B) {
  return __spirv_ArbitraryFloatGTINTEL<1 + EA + MA, 1 + EB + MB>(A, MA, B, MB);
  // CHECK: call spir_func zeroext i1 @_Z{{[0-9]+}}__spirv_ArbitraryFloatGTINTEL{{.*}}(i63 {{[%A-Za-z0-9.]+}}, i32 42, i63 {{[%A-Za-z0-9.]+}}, i32 41)
}
template auto ap_float_gt<20, 42, 21, 41>(sycl::detail::ap_int<1 + 20 + 42> A,
                                          sycl::detail::ap_int<1 + 21 + 41> B);

template <int EA, int MA, int EB, int MB>
SYCL_EXTERNAL auto ap_float_ge(sycl::detail::ap_int<1 + EA + MA> A,
                               sycl::detail::ap_int<1 + EB + MB> B) {
  return __spirv_ArbitraryFloatGEINTEL<1 + EA + MA, 1 + EB + MB>(A, MA, B, MB);
  // CHECK: call spir_func zeroext i1 @_Z{{[0-9]+}}__spirv_ArbitraryFloatGEINTEL{{.*}}(i47 {{[%A-Za-z0-9.]+}}, i32 27, i47 {{[%A-Za-z0-9.]+}}, i32 27)
}
template auto ap_float_ge<19, 27, 19, 27>(sycl::detail::ap_int<1 + 19 + 27> A,
                                          sycl::detail::ap_int<1 + 19 + 27> B);

template <int EA, int MA, int EB, int MB>
SYCL_EXTERNAL auto ap_float_lt(sycl::detail::ap_int<1 + EA + MA> A,
                               sycl::detail::ap_int<1 + EB + MB> B) {
  return __spirv_ArbitraryFloatLTINTEL<1 + EA + MA, 1 + EB + MB>(A, MA, B, MB);
  // CHECK: call spir_func zeroext i1 @_Z{{[0-9]+}}__spirv_ArbitraryFloatLTINTEL{{.*}}(i5 signext {{[%A-Za-z0-9.]+}}, i32 2, i7 signext {{[%A-Za-z0-9.]+}}, i32 3)
}
template auto ap_float_lt<2, 2, 3, 3>(sycl::detail::ap_int<1 + 2 + 2> A,
                                      sycl::detail::ap_int<1 + 3 + 3> B);

template <int EA, int MA, int EB, int MB>
SYCL_EXTERNAL auto ap_float_le(sycl::detail::ap_int<1 + EA + MA> A,
                               sycl::detail::ap_int<1 + EB + MB> B) {
  return __spirv_ArbitraryFloatLEINTEL<1 + EA + MA, 1 + EB + MB>(A, MA, B, MB);
  // CHECK: call spir_func zeroext i1 @_Z{{[0-9]+}}__spirv_ArbitraryFloatLEINTEL{{.*}}(i55 {{[%A-Za-z0-9.]+}}, i32 27, i55 {{[%A-Za-z0-9.]+}}, i32 28)
}
template auto ap_float_le<27, 27, 26, 28>(sycl::detail::ap_int<1 + 27 + 27> A,
                                          sycl::detail::ap_int<1 + 26 + 28> B);

template <int EA, int MA, int EB, int MB>
SYCL_EXTERNAL auto ap_float_eq(sycl::detail::ap_int<1 + EA + MA> A,
                               sycl::detail::ap_int<1 + EB + MB> B) {
  return __spirv_ArbitraryFloatEQINTEL<1 + EA + MA, 1 + EB + MB>(A, MA, B, MB);
  // CHECK: call spir_func zeroext i1 @_Z{{[0-9]+}}__spirv_ArbitraryFloatEQINTEL{{.*}}(i20 signext {{[%A-Za-z0-9.]+}}, i32 12, i15 signext {{[%A-Za-z0-9.]+}}, i32 7)
}
template auto ap_float_eq<7, 12, 7, 7>(sycl::detail::ap_int<1 + 7 + 12> A,
                                       sycl::detail::ap_int<1 + 7 + 7> B);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_recip(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatRecipINTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i39 @_Z{{[0-9]+}}__spirv_ArbitraryFloatRecipINTEL{{.*}}(i39 {{[%A-Za-z0-9.]+}}, i32 29, i32 29, i32 0, i32 2, i32 1)
}
template auto ap_float_recip<9, 29, 9, 29>(sycl::detail::ap_int<1 + 9 + 29> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_rsqrt(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatRSqrtINTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i34 @_Z{{[0-9]+}}__spirv_ArbitraryFloatRSqrtINTEL{{.*}}(i32 {{[%A-Za-z0-9.]+}}, i32 19, i32 20, i32 0, i32 2, i32 1)
}
template auto
ap_float_rsqrt<12, 19, 13, 20>(sycl::detail::ap_int<1 + 12 + 19> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_cbrt(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatCbrtINTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i2 @_Z{{[0-9]+}}__spirv_ArbitraryFloatCbrtINTEL{{.*}}(i2 signext {{[%A-Za-z0-9.]+}}, i32 1, i32 1, i32 0, i32 2, i32 1)
}
template auto ap_float_cbrt<0, 1, 0, 1>(sycl::detail::ap_int<1 + 0 + 1> A);

template <int EA, int MA, int EB, int MB, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_hypot(sycl::detail::ap_int<1 + EA + MA> A,
                                  sycl::detail::ap_int<1 + EB + MB> B) {
  return __spirv_ArbitraryFloatHypotINTEL<1 + EA + MA, 1 + EB + MB,
                                          1 + Eout + Mout>(
      A, MA, B, MB, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i42 @_Z{{[0-9]+}}__spirv_ArbitraryFloatHypotINTEL{{.*}}(i41 {{[%A-Za-z0-9.]+}}, i32 20, i43 {{[%A-Za-z0-9.]+}}, i32 21, i32 22, i32 0, i32 2, i32 1)
}
template auto
ap_float_hypot<20, 20, 21, 21, 19, 22>(sycl::detail::ap_int<1 + 20 + 20> A,
                                       sycl::detail::ap_int<1 + 21 + 21> B);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_sqrt(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatSqrtINTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i17 @_Z{{[0-9]+}}__spirv_ArbitraryFloatSqrtINTEL{{.*}}(i15 signext {{[%A-Za-z0-9.]+}}, i32 7, i32 8, i32 0, i32 2, i32 1)
}
template auto ap_float_sqrt<7, 7, 8, 8>(sycl::detail::ap_int<1 + 7 + 7> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_log(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatLogINTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i50 @_Z{{[0-9]+}}__spirv_ArbitraryFloatLogINTEL{{.*}}(i50 {{[%A-Za-z0-9.]+}}, i32 19, i32 30, i32 0, i32 2, i32 1)
}
template auto ap_float_log<30, 19, 19, 30>(sycl::detail::ap_int<1 + 30 + 19> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_log2(sycl::detail::ap_int<1 + EA + MA> A) {
  sycl::detail::ap_int<1 + Eout + Mout> log2_res =
      __spirv_ArbitraryFloatLog2INTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i38 @_Z{{[0-9]+}}__spirv_ArbitraryFloatLog2INTEL{{.*}}(i38 {{[%A-Za-z0-9.]+}}, i32 20, i32 19, i32 0, i32 2, i32 1)
}
template auto
ap_float_log2<17, 20, 18, 19>(sycl::detail::ap_int<1 + 17 + 20> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_log10(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatLog10INTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i10 @_Z{{[0-9]+}}__spirv_ArbitraryFloatLog10INTEL{{.*}}(i8 signext {{[%A-Za-z0-9.]+}}, i32 3, i32 5, i32 0, i32 2, i32 1)
}
template auto ap_float_log10<4, 3, 4, 5>(sycl::detail::ap_int<1 + 4 + 3> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_log1p(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatLog1pINTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i49 @_Z{{[0-9]+}}__spirv_ArbitraryFloatLog1pINTEL{{.*}}(i48 {{[%A-Za-z0-9.]+}}, i32 30, i32 30, i32 0, i32 2, i32 1)
}
template auto
ap_float_log1p<17, 30, 18, 30>(sycl::detail::ap_int<1 + 17 + 30> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_exp(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatExpINTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i42 @_Z{{[0-9]+}}__spirv_ArbitraryFloatExpINTEL{{.*}}(i42 {{[%A-Za-z0-9.]+}}, i32 25, i32 25, i32 0, i32 2, i32 1)
}
template auto ap_float_exp<16, 25, 16, 25>(sycl::detail::ap_int<1 + 16 + 25> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_exp2(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatExp2INTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i5 @_Z{{[0-9]+}}__spirv_ArbitraryFloatExp2INTEL{{.*}}(i3 signext {{[%A-Za-z0-9.]+}}, i32 1, i32 2, i32 0, i32 2, i32 1)
}
template auto ap_float_exp2<1, 1, 2, 2>(sycl::detail::ap_int<1 + 1 + 1> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_exp10(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatExp10INTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i25 @_Z{{[0-9]+}}__spirv_ArbitraryFloatExp10INTEL{{.*}}(i25 signext {{[%A-Za-z0-9.]+}}, i32 16, i32 16, i32 0, i32 2, i32 1)
}
template auto ap_float_exp10<8, 16, 8, 16>(sycl::detail::ap_int<1 + 8 + 16> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_expm1(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatExpm1INTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i62 @_Z{{[0-9]+}}__spirv_ArbitraryFloatExpm1INTEL{{.*}}(i64 {{[%A-Za-z0-9.]+}}, i32 42, i32 41, i32 0, i32 2, i32 1)
}
template auto
ap_float_expm1<21, 42, 20, 41>(sycl::detail::ap_int<1 + 21 + 42> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_sin(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatSinINTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i34 @_Z{{[0-9]+}}__spirv_ArbitraryFloatSinINTEL{{.*}}(i30 signext {{[%A-Za-z0-9.]+}}, i32 15, i32 17, i32 0, i32 2, i32 1)
}
template auto ap_float_sin<14, 15, 16, 17>(sycl::detail::ap_int<1 + 14 + 15> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_cos(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatCosINTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i4 @_Z{{[0-9]+}}__spirv_ArbitraryFloatCosINTEL{{.*}}(i4 signext {{[%A-Za-z0-9.]+}}, i32 2, i32 1, i32 0, i32 2, i32 1)
}
template auto ap_float_cos<1, 2, 2, 1>(sycl::detail::ap_int<1 + 1 + 2> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_sincos(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatSinCosINTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i62 @_Z{{[0-9]+}}__spirv_ArbitraryFloatSinCosINTEL{{.*}}(i27 signext {{[%A-Za-z0-9.]+}}, i32 18, i32 20, i32 0, i32 2, i32 1)
}
template auto
ap_float_sincos<8, 18, 10, 20>(sycl::detail::ap_int<1 + 8 + 18> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_sinpi(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatSinPiINTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i13 @_Z{{[0-9]+}}__spirv_ArbitraryFloatSinPiINTEL{{.*}}(i10 signext {{[%A-Za-z0-9.]+}}, i32 6, i32 6, i32 0, i32 2, i32 1)
}
template auto ap_float_sinpi<3, 6, 6, 6>(sycl::detail::ap_int<1 + 3 + 6> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_cospi(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatCosPiINTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i59 @_Z{{[0-9]+}}__spirv_ArbitraryFloatCosPiINTEL{{.*}}(i59 {{[%A-Za-z0-9.]+}}, i32 40, i32 40, i32 0, i32 2, i32 1)
}
template auto
ap_float_cospi<18, 40, 18, 40>(sycl::detail::ap_int<1 + 18 + 40> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_sincospi(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatSinCosPiINTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i64 @_Z{{[0-9]+}}__spirv_ArbitraryFloatSinCosPiINTEL{{.*}}(i30 signext {{[%A-Za-z0-9.]+}}, i32 20, i32 20, i32 0, i32 2, i32 1)
}
template auto
ap_float_sincospi<9, 20, 11, 20>(sycl::detail::ap_int<1 + 9 + 20> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_asin(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatASinINTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i11 @_Z{{[0-9]+}}__spirv_ArbitraryFloatASinINTEL{{.*}}(i7 signext {{[%A-Za-z0-9.]+}}, i32 4, i32 8, i32 0, i32 2, i32 1)
}
template auto ap_float_asin<2, 4, 2, 8>(sycl::detail::ap_int<1 + 2 + 4> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_asinpi(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatASinPiINTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i35 @_Z{{[0-9]+}}__spirv_ArbitraryFloatASinPiINTEL{{.*}}(i35 {{[%A-Za-z0-9.]+}}, i32 23, i32 23, i32 0, i32 2, i32 1)
}
template auto
ap_float_asinpi<11, 23, 11, 23>(sycl::detail::ap_int<1 + 11 + 23> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_acos(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatACosINTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i14 @_Z{{[0-9]+}}__spirv_ArbitraryFloatACosINTEL{{.*}}(i14 signext {{[%A-Za-z0-9.]+}}, i32 9, i32 10, i32 0, i32 2, i32 1)
}
template auto ap_float_acos<4, 9, 3, 10>(sycl::detail::ap_int<1 + 4 + 9> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_acospi(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatACosPiINTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i8 @_Z{{[0-9]+}}__spirv_ArbitraryFloatACosPiINTEL{{.*}}(i8 signext {{[%A-Za-z0-9.]+}}, i32 5, i32 4, i32 0, i32 2, i32 1)
}
template auto ap_float_acospi<2, 5, 3, 4>(sycl::detail::ap_int<1 + 2 + 5> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_atan(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatATanINTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i44 @_Z{{[0-9]+}}__spirv_ArbitraryFloatATanINTEL{{.*}}(i44 {{[%A-Za-z0-9.]+}}, i32 31, i32 31, i32 0, i32 2, i32 1)
}
template auto
ap_float_atan<12, 31, 12, 31>(sycl::detail::ap_int<1 + 12 + 31> A);

template <int EA, int MA, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_atapin(sycl::detail::ap_int<1 + EA + MA> A) {
  return __spirv_ArbitraryFloatATanPiINTEL<1 + EA + MA, 1 + Eout + Mout>(
      A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i34 @_Z{{[0-9]+}}__spirv_ArbitraryFloatATanPiINTEL{{.*}}(i40 {{[%A-Za-z0-9.]+}}, i32 38, i32 32, i32 0, i32 2, i32 1)
}
template auto ap_float_atapin<1, 38, 1, 32>(sycl::detail::ap_int<1 + 1 + 38> A);

template <int EA, int MA, int EB, int MB, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_atan2(sycl::detail::ap_int<1 + EA + MA> A,
                                  sycl::detail::ap_int<1 + EB + MB> B) {
  return __spirv_ArbitraryFloatATan2INTEL<1 + EA + MA, 1 + EB + MB,
                                          1 + Eout + Mout>(
      A, MA, B, MB, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i27 @_Z{{[0-9]+}}__spirv_ArbitraryFloatATan2INTEL{{.*}}(i24 signext {{[%A-Za-z0-9.]+}}, i32 16, i25 signext {{[%A-Za-z0-9.]+}}, i32 17, i32 18, i32 0, i32 2, i32 1)
}
template auto
ap_float_atan2<7, 16, 7, 17, 8, 18>(sycl::detail::ap_int<1 + 7 + 16> A,
                                    sycl::detail::ap_int<1 + 7 + 17> B);

template <int EA, int MA, int EB, int MB, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_pow(sycl::detail::ap_int<1 + EA + MA> A,
                                sycl::detail::ap_int<1 + EB + MB> B) {
  return __spirv_ArbitraryFloatPowINTEL<1 + EA + MA, 1 + EB + MB,
                                        1 + Eout + Mout>(
      A, MA, B, MB, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i21 @_Z{{[0-9]+}}__spirv_ArbitraryFloatPowINTEL{{.*}}(i17 signext {{[%A-Za-z0-9.]+}}, i32 8, i19 signext {{[%A-Za-z0-9.]+}}, i32 9, i32 10, i32 0, i32 2, i32 1)
}
template auto
ap_float_pow<8, 8, 9, 9, 10, 10>(sycl::detail::ap_int<1 + 8 + 8> A,
                                 sycl::detail::ap_int<1 + 9 + 9> B);

template <int EA, int MA, int EB, int MB, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_powr(sycl::detail::ap_int<1 + EA + MA> A,
                                 sycl::detail::ap_int<1 + EB + MB> B) {
  return __spirv_ArbitraryFloatPowRINTEL<1 + EA + MA, 1 + EB + MB,
                                         1 + Eout + Mout>(
      A, MA, B, MB, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i56 @_Z{{[0-9]+}}__spirv_ArbitraryFloatPowRINTEL{{.*}}(i54 {{[%A-Za-z0-9.]+}}, i32 35, i55 {{[%A-Za-z0-9.]+}}, i32 35, i32 35, i32 0, i32 2, i32 1)
}
template auto
ap_float_powr<18, 35, 19, 35, 20, 35>(sycl::detail::ap_int<1 + 18 + 35> A,
                                      sycl::detail::ap_int<1 + 19 + 35> B);

template <int EA, int MA, int WB, int Eout, int Mout>
SYCL_EXTERNAL auto ap_float_pown(sycl::detail::ap_int<1 + EA + MA> A,
                                 sycl::detail::ap_int<WB> B) {
  return __spirv_ArbitraryFloatPowNINTEL<1 + EA + MA, WB, 1 + Eout + Mout>(
      A, MA, B, SignOfB, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i15 @_Z{{[0-9]+}}__spirv_ArbitraryFloatPowNINTEL{{.*}}(i12 signext {{[%A-Za-z0-9.]+}}, i32 7, i10 signext {{[%A-Za-z0-9.]+}}, i1 zeroext false, i32 9, i32 0, i32 2, i32 1)
}
template auto ap_float_pown<4, 7, 10, 5, 9>(sycl::detail::ap_int<1 + 4 + 7> A,
                                            sycl::detail::ap_int<10> B);