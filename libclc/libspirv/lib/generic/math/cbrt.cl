//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#include <clc/math/tables.h>
#include <clc/clcmacro.h>
#include <clc/math/math.h>

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_cbrt(float x) {

  uint xi = __clc_as_uint(x);
  uint axi = xi & EXSIGNBIT_SP32;
  uint xsign = axi ^ xi;
  xi = axi;

  int m = (xi >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;

  // Treat subnormals
  uint xisub = __clc_as_uint(__clc_as_float(xi | 0x3f800000) - 1.0f);
  int msub = (xisub >> EXPSHIFTBITS_SP32) - 253;
  int c = m == -127;
  xi = c ? xisub : xi;
  m = c ? msub : m;

  int m3 = m / 3;
  int rem = m - m3 * 3;
  float mf = __clc_as_float((m3 + EXPBIAS_SP32) << EXPSHIFTBITS_SP32);

  uint indx = (xi & 0x007f0000) + ((xi & 0x00008000) << 1);
  float f =
      __clc_as_float((xi & MANTBITS_SP32) | 0x3f000000) - __clc_as_float(indx | 0x3f000000);

  indx >>= 16;
  float r = f * USE_TABLE(log_inv_tbl, indx);
  float poly =
      __spirv_ocl_mad(__spirv_ocl_mad(r, 0x1.f9add4p-5f, -0x1.c71c72p-4f),
                      r * r, r * 0x1.555556p-2f);

  // This could also be done with a 5-element table
  float remH = 0x1.428000p-1f;
  float remT = 0x1.45f31ap-14f;

  remH = rem == -1 ? 0x1.964000p-1f : remH;
  remT = rem == -1 ? 0x1.fea53ep-13f : remT;

  remH = rem == 0 ? 0x1.000000p+0f : remH;
  remT = rem == 0 ? 0x0.000000p+0f : remT;

  remH = rem == 1 ? 0x1.428000p+0f : remH;
  remT = rem == 1 ? 0x1.45f31ap-13f : remT;

  remH = rem == 2 ? 0x1.964000p+0f : remH;
  remT = rem == 2 ? 0x1.fea53ep-12f : remT;

  float2 tv = USE_TABLE(cbrt_tbl, indx);
  float cbrtH = tv.s0;
  float cbrtT = tv.s1;

  float bH = cbrtH * remH;
  float bT =
      __spirv_ocl_mad(cbrtH, remT, __spirv_ocl_mad(cbrtT, remH, cbrtT * remT));

  float z = __spirv_ocl_mad(poly, bH, __spirv_ocl_mad(poly, bT, bT)) + bH;
  z *= mf;
  z = __clc_as_float(__clc_as_uint(z) | xsign);
  c = axi >= EXPBITS_SP32 | axi == 0;
  z = c ? x : z;
  return z;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_cbrt, float);

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_cbrt(double x) {

  int return_x = __spirv_IsInf(x) | __spirv_IsNan(x) | x == 0.0;
  ulong ux = __clc_as_ulong(__spirv_ocl_fabs(x));
  int m = (__clc_as_int2(ux).hi >> 20) - 1023;

  // Treat subnormals
  ulong uxs = __clc_as_ulong(__clc_as_double(0x3ff0000000000000UL | ux) - 1.0);
  int ms = m + (__clc_as_int2(uxs).hi >> 20) - 1022;

  int c = m == -1023;
  ux = c ? uxs : ux;
  m = c ? ms : m;

  int mby3 = m / 3;
  int rem = m - 3 * mby3;

  double mf = __clc_as_double((ulong)(mby3 + 1023) << 52);

  ux &= 0x000fffffffffffffUL;
  double Y = __clc_as_double(0x3fe0000000000000UL | ux);

  // nearest integer
  int index = __clc_as_int2(ux).hi >> 11;
  index = (0x100 | (index >> 1)) + (index & 1);
  double F = (double)index * 0x1.0p-9;

  double f = Y - F;
  double r = f * USE_TABLE(cbrt_inv_tbl, index - 256);

  double z =
      r * __spirv_ocl_fma(
              r,
              __spirv_ocl_fma(
                  r,
                  __spirv_ocl_fma(
                      r,
                      __spirv_ocl_fma(r,
                                      __spirv_ocl_fma(r, -0x1.8090d6221a247p-6,
                                                      0x1.ee7113506ac13p-6),
                                      -0x1.511e8d2b3183bp-5),
                      0x1.f9add3c0ca458p-5),
                  -0x1.c71c71c71c71cp-4),
              0x1.5555555555555p-2);

  double2 tv = USE_TABLE(cbrt_rem_tbl, rem + 2);
  double Rem_h = tv.s0;
  double Rem_t = tv.s1;

  tv = USE_TABLE(cbrt_dbl_tbl, index - 256);
  double F_h = tv.s0;
  double F_t = tv.s1;

  double b_h = F_h * Rem_h;
  double b_t =
      __spirv_ocl_fma(Rem_t, F_h, __spirv_ocl_fma(F_t, Rem_h, F_t * Rem_t));

  double ans = __spirv_ocl_fma(z, b_h, __spirv_ocl_fma(z, b_t, b_t)) + b_h;
  ans = __spirv_ocl_copysign(ans * mf, x);
  return return_x ? x : ans;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_cbrt, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_UNARY_BUILTIN_SCALARIZE(half, __spirv_ocl_cbrt, __builtin_cbrtf, half)

#endif
