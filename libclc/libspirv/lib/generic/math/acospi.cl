//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#include <clc/clcmacro.h>
#include <clc/math/math.h>

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_acospi(float x) {
  // Computes arccos(x).
  // The argument is first reduced by noting that arccos(x)
  // is invalid for abs(x) > 1. For denormal and small
  // arguments arccos(x) = pi/2 to machine accuracy.
  // Remaining argument ranges are handled as follows.
  // For abs(x) <= 0.5 use
  // arccos(x) = pi/2 - arcsin(x)
  // = pi/2 - (x + x^3*R(x^2))
  // where R(x^2) is a rational minimax approximation to
  // (arcsin(x) - x)/x^3.
  // For abs(x) > 0.5 exploit the identity:
  // arccos(x) = pi - 2*arcsin(sqrt(1-x)/2)
  // together with the above rational approximation, and
  // reconstruct the terms carefully.

  // Some constants and split constants.
  const float pi = 3.1415926535897933e+00f;
  const float piby2_head = 1.5707963267948965580e+00f;  /* 0x3ff921fb54442d18 */
  const float piby2_tail = 6.12323399573676603587e-17f; /* 0x3c91a62633145c07 */

  uint ux = __clc_as_uint(x);
  uint aux = ux & ~SIGNBIT_SP32;
  int xneg = ux != aux;
  int xexp = (int)(aux >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;

  float y = __clc_as_float(aux);

  // transform if |x| >= 0.5
  int transform = xexp >= -1;

  float y2 = y * y;
  float yt = 0.5f * (1.0f - y);
  float r = transform ? yt : y2;

  // Use a rational approximation for [0.0, 0.5]
  float a = __spirv_ocl_mad(
      r,
      __spirv_ocl_mad(r,
                      __spirv_ocl_mad(r, -0.00396137437848476485201154797087F,
                                      -0.0133819288943925804214011424456F),
                      -0.0565298683201845211985026327361F),
      0.184161606965100694821398249421F);
  float b = __spirv_ocl_mad(r, -0.836411276854206731913362287293F,
                            1.10496961524520294485512696706F);
  float u = r * MATH_DIVIDE(a, b);

  float s = __spirv_ocl_sqrt(r);
  y = s;
  float s1 = __clc_as_float(__clc_as_uint(s) & 0xffff0000);
  float c = MATH_DIVIDE(r - s1 * s1, s + s1);
  // float rettn = 1.0f - MATH_DIVIDE(2.0f * (s + (y * u - piby2_tail)), pi);
  float rettn =
      1.0f - MATH_DIVIDE(2.0f * (s + __spirv_ocl_mad(y, u, -piby2_tail)), pi);
  // float rettp = MATH_DIVIDE(2.0F * s1 + (2.0F * c + 2.0F * y * u), pi);
  float rettp = MATH_DIVIDE(2.0f * (s1 + __spirv_ocl_mad(y, u, c)), pi);
  float rett = xneg ? rettn : rettp;
  // float ret = MATH_DIVIDE(piby2_head - (x - (piby2_tail - x * u)), pi);
  float ret =
      MATH_DIVIDE(piby2_head - (x - __spirv_ocl_mad(x, -u, piby2_tail)), pi);

  ret = transform ? rett : ret;
  ret = aux > 0x3f800000U ? __clc_as_float(QNANBITPATT_SP32) : ret;
  ret = ux == 0x3f800000U ? 0.0f : ret;
  ret = ux == 0xbf800000U ? 1.0f : ret;
  ret = xexp < -26 ? 0.5f : ret;
  return ret;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_acospi, float)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_acospi(double x) {
  // Computes arccos(x).
  // The argument is first reduced by noting that arccos(x)
  // is invalid for abs(x) > 1. For denormal and small
  // arguments arccos(x) = pi/2 to machine accuracy.
  // Remaining argument ranges are handled as follows.
  // For abs(x) <= 0.5 use
  // arccos(x) = pi/2 - arcsin(x)
  // = pi/2 - (x + x^3*R(x^2))
  // where R(x^2) is a rational minimax approximation to
  // (arcsin(x) - x)/x^3.
  // For abs(x) > 0.5 exploit the identity:
  // arccos(x) = pi - 2*arcsin(sqrt(1-x)/2)
  // together with the above rational approximation, and
  // reconstruct the terms carefully.

  const double pi = 0x1.921fb54442d18p+1;
  const double piby2_tail = 6.12323399573676603587e-17; /* 0x3c91a62633145c07 */

  double y = __spirv_ocl_fabs(x);
  int xneg = __clc_as_int2(x).hi < 0;
  int xexp = (__clc_as_int2(y).hi >> 20) - EXPBIAS_DP64;

  // abs(x) >= 0.5
  int transform = xexp >= -1;

  // Transform y into the range [0,0.5)
  double r1 = 0.5 * (1.0 - y);
  double s = __spirv_ocl_sqrt(r1);
  double r = y * y;
  r = transform ? r1 : r;
  y = transform ? s : y;

  // Use a rational approximation for [0.0, 0.5]
  double un = __spirv_ocl_fma(
      r,
      __spirv_ocl_fma(
          r,
          __spirv_ocl_fma(
              r,
              __spirv_ocl_fma(
                  r,
                  __spirv_ocl_fma(r, 0.0000482901920344786991880522822991,
                                  0.00109242697235074662306043804220),
                  -0.0549989809235685841612020091328),
              0.275558175256937652532686256258),
          -0.445017216867635649900123110649),
      0.227485835556935010735943483075);

  double ud = __spirv_ocl_fma(
      r,
      __spirv_ocl_fma(
          r,
          __spirv_ocl_fma(r,
                          __spirv_ocl_fma(r, 0.105869422087204370341222318533,
                                          -0.943639137032492685763471240072),
                          2.76568859157270989520376345954),
          -3.28431505720958658909889444194),
      1.36491501334161032038194214209);

  double u = r * MATH_DIVIDE(un, ud);

  // Reconstruct acos carefully in transformed region
  double res1 = __spirv_ocl_fma(
      -2.0, MATH_DIVIDE(s + __spirv_ocl_fma(y, u, -piby2_tail), pi), 1.0);
  double s1 = __clc_as_double(__clc_as_ulong(s) & 0xffffffff00000000UL);
  double c = MATH_DIVIDE(__spirv_ocl_fma(-s1, s1, r), s + s1);
  double res2 = MATH_DIVIDE(
      __spirv_ocl_fma(2.0, s1, __spirv_ocl_fma(2.0, c, 2.0 * y * u)), pi);
  res1 = xneg ? res1 : res2;
  res2 = 0.5 - __spirv_ocl_fma(x, u, x) / pi;
  res1 = transform ? res1 : res2;

  const double qnan = __clc_as_double(QNANBITPATT_DP64);
  res2 = x == 1.0 ? 0.0 : qnan;
  res2 = x == -1.0 ? 1.0 : res2;
  res1 = xexp >= 0 ? res2 : res1;
  res1 = xexp < -56 ? 0.5 : res1;

  return res1;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_acospi, double)

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half __spirv_ocl_acospi(half x) {
  float t = x;
  return __spirv_ocl_acospi(t);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __spirv_ocl_acospi, half)

#endif
