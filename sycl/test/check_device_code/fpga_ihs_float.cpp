//==- fpga_ihs_float.cpp - SYCL FPGA arbitrary precision floating point test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -I %sycl_include -S -emit-llvm -fsycl -fsycl-device-only %s -o - | FileCheck %s
// RUN: %clangxx -I %sycl_include -S -emit-llvm -fsycl -fno-sycl-early-optimizations -fsycl-device-only %s -o - | FileCheck %s

#include "CL/__spirv/spirv_ops.hpp"

constexpr int32_t Subnorm = 0;
constexpr int32_t RndMode = 2;
constexpr int32_t RndAcc = 1;
constexpr bool FromSign = false;
constexpr bool ToSign = true;

template <int EA, int MA, int Eout, int Mout>
void ap_float_cast() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> float_cast_res =
      __spirv_ArbitraryFloatCastINTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i40 @_Z{{[0-9]+}}__spirv_ArbitraryFloatCastINTEL{{.*}}(i40 {{[%a-z0-9.]+}}, i32 28, i32 30, i32 0, i32 2, i32 1)
}

template <int WA, int Eout, int Mout>
void ap_float_cast_from_int() {
  ap_int<WA> A;
  ap_int<1 + Eout + Mout> cast_from_int_res =
      __spirv_ArbitraryFloatCastFromIntINTEL<WA, 1 + Eout + Mout>(
          A, Mout, FromSign, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i25 @_Z{{[0-9]+}}__spirv_ArbitraryFloatCastFromIntINTEL{{.*}}(i43 {{[%a-z0-9.]+}}, i32 16, i1 zeroext false, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Wout>
void ap_float_cast_to_int() {
  ap_int<1 + EA + MA> A;
  ap_int<Wout> cast_to_int_res =
      __spirv_ArbitraryFloatCastToIntINTEL<1 + EA + MA, Wout>(
          A, MA, ToSign, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i30 @_Z{{[0-9]+}}__spirv_ArbitraryFloatCastToIntINTEL{{.*}}(i23 signext {{[%a-z0-9.]+}}, i32 15, i1 zeroext true, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int EB, int MB, int Eout, int Mout>
void ap_float_add() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + EB + MB> B;
  ap_int<1 + Eout + Mout> add_res =
      __spirv_ArbitraryFloatAddINTEL<1 + EA + MA, 1 + EB + MB, 1 + Eout + Mout>(
          A, MA, B, MB, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i14 @_Z{{[0-9]+}}__spirv_ArbitraryFloatAddINTEL{{.*}}(i13 signext {{[%a-z0-9.]+}}, i32 7, i15 signext {{[%a-z0-9.]+}}, i32 8, i32 9, i32 0, i32 2, i32 1)
  // CHECK: call spir_func signext i13 @_Z{{[0-9]+}}__spirv_ArbitraryFloatAddINTEL{{.*}}(i15 signext {{[%a-z0-9.]+}}, i32 8, i14 signext {{[%a-z0-9.]+}}, i32 9, i32 7, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int EB, int MB, int Eout, int Mout>
void ap_float_sub() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + EB + MB> B;
  ap_int<1 + Eout + Mout> sub_res =
      __spirv_ArbitraryFloatSubINTEL<1 + EA + MA, 1 + EB + MB, 1 + Eout + Mout>(
          A, MA, B, MB, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i13 @_Z{{[0-9]+}}__spirv_ArbitraryFloatSubINTEL{{.*}}(i9 signext {{[%a-z0-9.]+}}, i32 4, i11 signext {{[%a-z0-9.]+}}, i32 5, i32 6, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int EB, int MB, int Eout, int Mout>
void ap_float_mul() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + EB + MB> B;
  ap_int<1 + Eout + Mout> mul_res =
      __spirv_ArbitraryFloatMulINTEL<1 + EA + MA, 1 + EB + MB, 1 + Eout + Mout>(
          A, MA, B, MB, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i51 @_Z{{[0-9]+}}__spirv_ArbitraryFloatMulINTEL{{.*}}(i51 {{[%a-z0-9.]+}}, i32 34, i51 {{[%a-z0-9.]+}}, i32 34, i32 34, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int EB, int MB, int Eout, int Mout>
void ap_float_div() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + EB + MB> B;
  ap_int<1 + Eout + Mout> div_res =
      __spirv_ArbitraryFloatDivINTEL<1 + EA + MA, 1 + EB + MB, 1 + Eout + Mout>(
          A, MA, B, MB, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i18 @_Z{{[0-9]+}}__spirv_ArbitraryFloatDivINTEL{{.*}}(i16 signext {{[%a-z0-9.]+}}, i32 11, i16 signext {{[%a-z0-9.]+}}, i32 11, i32 12, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int EB, int MB>
void ap_float_gt() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + EB + MB> B;
  bool gt_res = __spirv_ArbitraryFloatGTINTEL<1 + EA + MA, 1 + EB + MB>(A, MA, B, MB);
  // CHECK: call spir_func zeroext i1 @_Z{{[0-9]+}}__spirv_ArbitraryFloatGTINTEL{{.*}}(i63 {{[%a-z0-9.]+}}, i32 42, i63 {{[%a-z0-9.]+}}, i32 41)
}

template <int EA, int MA, int EB, int MB>
void ap_float_ge() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + EB + MB> B;
  bool ge_res = __spirv_ArbitraryFloatGEINTEL<1 + EA + MA, 1 + EB + MB>(A, MA, B, MB);
  // CHECK: call spir_func zeroext i1 @_Z{{[0-9]+}}__spirv_ArbitraryFloatGEINTEL{{.*}}(i47 {{[%a-z0-9.]+}}, i32 27, i47 {{[%a-z0-9.]+}}, i32 27)
}

template <int EA, int MA, int EB, int MB>
void ap_float_lt() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + EB + MB> B;
  bool lt_res = __spirv_ArbitraryFloatLTINTEL<1 + EA + MA, 1 + EB + MB>(A, MA, B, MB);
  // CHECK: call spir_func zeroext i1 @_Z{{[0-9]+}}__spirv_ArbitraryFloatLTINTEL{{.*}}(i5 signext {{[%a-z0-9.]+}}, i32 2, i7 signext {{[%a-z0-9.]+}}, i32 3)
}

template <int EA, int MA, int EB, int MB>
void ap_float_le() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + EB + MB> B;
  bool le_res = __spirv_ArbitraryFloatLEINTEL<1 + EA + MA, 1 + EB + MB>(A, MA, B, MB);
  // CHECK: call spir_func zeroext i1 @_Z{{[0-9]+}}__spirv_ArbitraryFloatLEINTEL{{.*}}(i55 {{[%a-z0-9.]+}}, i32 27, i55 {{[%a-z0-9.]+}}, i32 28)
}

template <int EA, int MA, int EB, int MB>
void ap_float_eq() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + EB + MB> B;
  bool eq_res = __spirv_ArbitraryFloatEQINTEL<1 + EA + MA, 1 + EB + MB>(A, MA, B, MB);
  // CHECK: call spir_func zeroext i1 @_Z{{[0-9]+}}__spirv_ArbitraryFloatEQINTEL{{.*}}(i20 signext {{[%a-z0-9.]+}}, i32 12, i15 signext {{[%a-z0-9.]+}}, i32 7)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_recip() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> recip_res =
      __spirv_ArbitraryFloatRecipINTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i39 @_Z{{[0-9]+}}__spirv_ArbitraryFloatRecipINTEL{{.*}}(i39 {{[%a-z0-9.]+}}, i32 29, i32 29, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_rsqrt() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> rsqrt_res =
      __spirv_ArbitraryFloatRSqrtINTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i34 @_Z{{[0-9]+}}__spirv_ArbitraryFloatRSqrtINTEL{{.*}}(i32 {{[%a-z0-9.]+}}, i32 19, i32 20, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_cbrt() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> cbrt_res =
      __spirv_ArbitraryFloatCbrtINTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i2 @_Z{{[0-9]+}}__spirv_ArbitraryFloatCbrtINTEL{{.*}}(i2 signext {{[%a-z0-9.]+}}, i32 1, i32 1, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int EB, int MB, int Eout, int Mout>
void ap_float_hypot() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + EB + MB> B;
  ap_int<1 + Eout + Mout> hypot_res =
      __spirv_ArbitraryFloatHypotINTEL<1 + EA + MA, 1 + EB + MB, 1 + Eout + Mout>(
          A, MA, B, MB, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i42 @_Z{{[0-9]+}}__spirv_ArbitraryFloatHypotINTEL{{.*}}(i41 {{[%a-z0-9.]+}}, i32 20, i43 {{[%a-z0-9.]+}}, i32 21, i32 22, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_sqrt() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> sqrt_res =
      __spirv_ArbitraryFloatSqrtINTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i17 @_Z{{[0-9]+}}__spirv_ArbitraryFloatSqrtINTEL{{.*}}(i15 signext {{[%a-z0-9.]+}}, i32 7, i32 8, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_log() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> log_res =
      __spirv_ArbitraryFloatLogINTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i50 @_Z{{[0-9]+}}__spirv_ArbitraryFloatLogINTEL{{.*}}(i50 {{[%a-z0-9.]+}}, i32 19, i32 30, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_log2() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> log2_res =
      __spirv_ArbitraryFloatLog2INTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i38 @_Z{{[0-9]+}}__spirv_ArbitraryFloatLog2INTEL{{.*}}(i38 {{[%a-z0-9.]+}}, i32 20, i32 19, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_log10() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> log10_res =
      __spirv_ArbitraryFloatLog10INTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i10 @_Z{{[0-9]+}}__spirv_ArbitraryFloatLog10INTEL{{.*}}(i8 signext {{[%a-z0-9.]+}}, i32 3, i32 5, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_log1p() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> log1p_res =
      __spirv_ArbitraryFloatLog1pINTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i49 @_Z{{[0-9]+}}__spirv_ArbitraryFloatLog1pINTEL{{.*}}(i48 {{[%a-z0-9.]+}}, i32 30, i32 30, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_exp() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> exp_res =
      __spirv_ArbitraryFloatExpINTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i42 @_Z{{[0-9]+}}__spirv_ArbitraryFloatExpINTEL{{.*}}(i42 {{[%a-z0-9.]+}}, i32 25, i32 25, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_exp2() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> exp2_res =
      __spirv_ArbitraryFloatExp2INTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i5 @_Z{{[0-9]+}}__spirv_ArbitraryFloatExp2INTEL{{.*}}(i3 signext {{[%a-z0-9.]+}}, i32 1, i32 2, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_exp10() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> exp10_res =
      __spirv_ArbitraryFloatExp10INTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i25 @_Z{{[0-9]+}}__spirv_ArbitraryFloatExp10INTEL{{.*}}(i25 signext {{[%a-z0-9.]+}}, i32 16, i32 16, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_expm1() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> expm1_res =
      __spirv_ArbitraryFloatExpm1INTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i62 @_Z{{[0-9]+}}__spirv_ArbitraryFloatExpm1INTEL{{.*}}(i64 {{[%a-z0-9.]+}}, i32 42, i32 41, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_sin() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> sin_res =
      __spirv_ArbitraryFloatSinINTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i34 @_Z{{[0-9]+}}__spirv_ArbitraryFloatSinINTEL{{.*}}(i30 signext {{[%a-z0-9.]+}}, i32 15, i32 17, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_cos() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> cos_res =
      __spirv_ArbitraryFloatCosINTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i4 @_Z{{[0-9]+}}__spirv_ArbitraryFloatCosINTEL{{.*}}(i4 signext {{[%a-z0-9.]+}}, i32 2, i32 1, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_sincos() {
  ap_int<1 + EA + MA> A;
  ap_int<2 * (1 + Eout + Mout)> sincos_res =
      __spirv_ArbitraryFloatSinCosINTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i62 @_Z{{[0-9]+}}__spirv_ArbitraryFloatSinCosINTEL{{.*}}(i27 signext {{[%a-z0-9.]+}}, i32 18, i32 20, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_sinpi() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> sinpi_res =
      __spirv_ArbitraryFloatSinPiINTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i13 @_Z{{[0-9]+}}__spirv_ArbitraryFloatSinPiINTEL{{.*}}(i10 signext {{[%a-z0-9.]+}}, i32 6, i32 6, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_cospi() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> cospi_res =
      __spirv_ArbitraryFloatCosPiINTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i59 @_Z{{[0-9]+}}__spirv_ArbitraryFloatCosPiINTEL{{.*}}(i59 {{[%a-z0-9.]+}}, i32 40, i32 40, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_sincospi() {
  ap_int<1 + EA + MA> A;
  ap_int<2 * (1 + Eout + Mout)> sincos_res =
      __spirv_ArbitraryFloatSinCosPiINTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i64 @_Z{{[0-9]+}}__spirv_ArbitraryFloatSinCosPiINTEL{{.*}}(i30 signext {{[%a-z0-9.]+}}, i32 20, i32 20, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_asin() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> asin_res =
      __spirv_ArbitraryFloatASinINTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i11 @_Z{{[0-9]+}}__spirv_ArbitraryFloatASinINTEL{{.*}}(i7 signext {{[%a-z0-9.]+}}, i32 4, i32 8, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_asinpi() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> asinpi_res =
      __spirv_ArbitraryFloatASinPiINTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i35 @_Z{{[0-9]+}}__spirv_ArbitraryFloatASinPiINTEL{{.*}}(i35 {{[%a-z0-9.]+}}, i32 23, i32 23, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_acos() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> acos_res =
      __spirv_ArbitraryFloatACosINTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i14 @_Z{{[0-9]+}}__spirv_ArbitraryFloatACosINTEL{{.*}}(i14 signext {{[%a-z0-9.]+}}, i32 9, i32 10, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_acospi() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> acospi_res =
      __spirv_ArbitraryFloatACosPiINTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i8 @_Z{{[0-9]+}}__spirv_ArbitraryFloatACosPiINTEL{{.*}}(i8 signext {{[%a-z0-9.]+}}, i32 5, i32 4, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_atan() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> atan_res =
      __spirv_ArbitraryFloatATanINTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i44 @_Z{{[0-9]+}}__spirv_ArbitraryFloatATanINTEL{{.*}}(i44 {{[%a-z0-9.]+}}, i32 31, i32 31, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int Eout, int Mout>
void ap_float_atapin() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + Eout + Mout> atanpi_res =
      __spirv_ArbitraryFloatATanPiINTEL<1 + EA + MA, 1 + Eout + Mout>(
          A, MA, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i34 @_Z{{[0-9]+}}__spirv_ArbitraryFloatATanPiINTEL{{.*}}(i40 {{[%a-z0-9.]+}}, i32 38, i32 32, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int EB, int MB, int Eout, int Mout>
void ap_float_atan2() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + EB + MB> B;
  ap_int<1 + Eout + Mout> atan2_res =
      __spirv_ArbitraryFloatATan2INTEL<1 + EA + MA, 1 + EB + MB, 1 + Eout + Mout>(
          A, MA, B, MB, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i27 @_Z{{[0-9]+}}__spirv_ArbitraryFloatATan2INTEL{{.*}}(i24 signext {{[%a-z0-9.]+}}, i32 16, i25 signext {{[%a-z0-9.]+}}, i32 17, i32 18, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int EB, int MB, int Eout, int Mout>
void ap_float_pow() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + EB + MB> B;
  ap_int<1 + Eout + Mout> pow_res =
      __spirv_ArbitraryFloatPowINTEL<1 + EA + MA, 1 + EB + MB, 1 + Eout + Mout>(
          A, MA, B, MB, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i21 @_Z{{[0-9]+}}__spirv_ArbitraryFloatPowINTEL{{.*}}(i17 signext {{[%a-z0-9.]+}}, i32 8, i19 signext {{[%a-z0-9.]+}}, i32 9, i32 10, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int EB, int MB, int Eout, int Mout>
void ap_float_powr() {
  ap_int<1 + EA + MA> A;
  ap_int<1 + EB + MB> B;
  ap_int<1 + Eout + Mout> powr_res =
      __spirv_ArbitraryFloatPowRINTEL<1 + EA + MA, 1 + EB + MB, 1 + Eout + Mout>(
          A, MA, B, MB, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func i56 @_Z{{[0-9]+}}__spirv_ArbitraryFloatPowRINTEL{{.*}}(i54 {{[%a-z0-9.]+}}, i32 35, i55 {{[%a-z0-9.]+}}, i32 35, i32 35, i32 0, i32 2, i32 1)
}

template <int EA, int MA, int WB, int Eout, int Mout>
void ap_float_pown() {
  ap_int<1 + EA + MA> A;
  ap_int<WB> B;
  ap_int<1 + Eout + Mout> pown_res =
      __spirv_ArbitraryFloatPowNINTEL<1 + EA + MA, WB, 1 + Eout + Mout>(
          A, MA, B, Mout, Subnorm, RndMode, RndAcc);
  // CHECK: call spir_func signext i15 @_Z{{[0-9]+}}__spirv_ArbitraryFloatPowNINTEL{{.*}}(i12 signext {{[%a-z0-9.]+}}, i32 7, i10 signext {{[%a-z0-9.]+}}, i32 9, i32 0, i32 2, i32 1)
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    ap_float_cast<11, 28, 9, 30>();
    ap_float_cast_from_int<43, 8, 16>();
    ap_float_cast_to_int<7, 15, 30>();
    ap_float_add<5, 7, 6, 8, 4, 9>();
    ap_float_add<6, 8, 4, 9, 5, 7>();
    ap_float_sub<4, 4, 5, 5, 6, 6>();
    ap_float_mul<16, 34, 16, 34, 16, 34>();
    ap_float_div<4, 11, 4, 11, 5, 12>();
    ap_float_gt<20, 42, 21, 41>();
    ap_float_ge<19, 27, 19, 27>();
    ap_float_lt<2, 2, 3, 3>();
    ap_float_le<27, 27, 26, 28>();
    ap_float_eq<7, 12, 7, 7>();
    ap_float_recip<9, 29, 9, 29>();
    ap_float_rsqrt<12, 19, 13, 20>();
    ap_float_cbrt<0, 1, 0, 1>();
    ap_float_hypot<20, 20, 21, 21, 19, 22>();
    ap_float_sqrt<7, 7, 8, 8>();
    ap_float_log<30, 19, 19, 30>();
    ap_float_log2<17, 20, 18, 19>();
    ap_float_log10<4, 3, 4, 5>();
    ap_float_log1p<17, 30, 18, 30>();
    ap_float_exp<16, 25, 16, 25>();
    ap_float_exp2<1, 1, 2, 2>();
    ap_float_exp10<8, 16, 8, 16>();
    ap_float_expm1<21, 42, 20, 41>();
    ap_float_sin<14, 15, 16, 17>();
    ap_float_cos<1, 2, 2, 1>();
    ap_float_sincos<8, 18, 10, 20>();
    ap_float_sinpi<3, 6, 6, 6>();
    ap_float_cospi<18, 40, 18, 40>();
    ap_float_sincospi<9, 20, 11, 20>();
    ap_float_asin<2, 4, 2, 8>();
    ap_float_asinpi<11, 23, 11, 23>();
    ap_float_acos<4, 9, 3, 10>();
    ap_float_acospi<2, 5, 3, 4>();
    ap_float_atan<12, 31, 12, 31>();
    ap_float_atapin<1, 38, 1, 32>();
    ap_float_atan2<7, 16, 7, 17, 8, 18>();
    ap_float_pow<8, 8, 9, 9, 10, 10>();
    ap_float_powr<18, 35, 19, 35, 20, 35>();
    ap_float_pown<4, 7, 10, 5, 9>();
  });
  return 0;
}
