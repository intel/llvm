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

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_atan2(float y, float x) {
  const float pi = 0x1.921fb6p+1f;
  const float piby2 = 0x1.921fb6p+0f;
  const float piby4 = 0x1.921fb6p-1f;
  const float threepiby4 = 0x1.2d97c8p+1f;

  float ax = __spirv_ocl_fabs(x);
  float ay = __spirv_ocl_fabs(y);
  float v = __spirv_ocl_fmin(ax, ay);
  float u = __spirv_ocl_fmax(ax, ay);

  // Scale since u could be large, as in "regular" divide
  float s = u > 0x1.0p+96f ? 0x1.0p-32f : 1.0f;
  float vbyu = s * MATH_DIVIDE(v, s * u);

  float vbyu2 = vbyu * vbyu;

#define USE_2_2_APPROXIMATION
#if defined USE_2_2_APPROXIMATION
  float p = __spirv_ocl_mad(
                vbyu2, __spirv_ocl_mad(vbyu2, -0x1.7e1f78p-9f, -0x1.7d1b98p-3f),
                -0x1.5554d0p-2f) *
            vbyu2 * vbyu;
  float q = __spirv_ocl_mad(
      vbyu2, __spirv_ocl_mad(vbyu2, 0x1.1a714cp-2f, 0x1.287c56p+0f), 1.0f);
#else
  float p = __spirv_ocl_mad(
                vbyu2, __spirv_ocl_mad(vbyu2, -0x1.55cd22p-5f, -0x1.26cf76p-2f),
                -0x1.55554ep-2f) *
            vbyu2 * vbyu;
  float q = __spirv_ocl_mad(
      vbyu2,
      __spirv_ocl_mad(vbyu2,
                      __spirv_ocl_mad(vbyu2, 0x1.9f1304p-5f, 0x1.2656fap-1f),
                      0x1.76b4b8p+0f),
      1.0f);
#endif

  // Octant 0 result
  float a = __spirv_ocl_mad(p, MATH_RECIP(q), vbyu);

  // Fix up 3 other octants
  float at = piby2 - a;
  a = ay > ax ? at : a;
  at = pi - a;
  a = x < 0.0F ? at : a;

  // y == 0 => 0 for x >= 0, pi for x < 0
  at = __clc_as_int(x) < 0 ? pi : 0.0f;
  a = y == 0.0f ? at : a;

  // if (!FINITE_ONLY()) {
  // x and y are +- Inf
  at = x > 0.0f ? piby4 : threepiby4;
  a = ax == INFINITY && ay == INFINITY ? at : a;

  // x or y is NaN
  a = __spirv_IsNan(x) || __spirv_IsNan(y) ? __clc_as_float(QNANBITPATT_SP32) : a;
  // }

  // Fixup sign and return
  return __spirv_ocl_copysign(a, y);
}

_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_atan2, float,
                      float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_atan2(double y, double x) {
  const double pi = 3.1415926535897932e+00;          /* 0x400921fb54442d18 */
  const double piby2 = 1.5707963267948966e+00;       /* 0x3ff921fb54442d18 */
  const double piby4 = 7.8539816339744831e-01;       /* 0x3fe921fb54442d18 */
  const double three_piby4 = 2.3561944901923449e+00; /* 0x4002d97c7f3321d2 */
  const double pi_head = 3.1415926218032836e+00;     /* 0x400921fb50000000 */
  const double pi_tail = 3.1786509547056392e-08;     /* 0x3e6110b4611a6263 */
  const double piby2_head = 1.5707963267948965e+00;  /* 0x3ff921fb54442d18 */
  const double piby2_tail = 6.1232339957367660e-17;  /* 0x3c91a62633145c07 */

  double x2 = x;
  int xneg = __clc_as_int2(x).hi < 0;
  int xexp = (__clc_as_int2(x).hi >> 20) & 0x7ff;

  double y2 = y;
  int yneg = __clc_as_int2(y).hi < 0;
  int yexp = (__clc_as_int2(y).hi >> 20) & 0x7ff;

  int cond2 = (xexp < 1021) & (yexp < 1021);
  int diffexp = yexp - xexp;

  // Scale up both x and y if they are both below 1/4
  double x1 = __spirv_ocl_ldexp(x, 1024);
  int xexp1 = (__clc_as_int2(x1).hi >> 20) & 0x7ff;
  double y1 = __spirv_ocl_ldexp(y, 1024);
  int yexp1 = (__clc_as_int2(y1).hi >> 20) & 0x7ff;
  int diffexp1 = yexp1 - xexp1;

  diffexp = cond2 ? diffexp1 : diffexp;
  x = cond2 ? x1 : x;
  y = cond2 ? y1 : y;

  // General case: take absolute values of arguments
  double u = __spirv_ocl_fabs(x);
  double v = __spirv_ocl_fabs(y);

  // Swap u and v if necessary to obtain 0 < v < u. Compute v/u.
  int swap_vu = u < v;
  double uu = u;
  u = swap_vu ? v : u;
  v = swap_vu ? uu : v;

  double vbyu = v / u;
  double q1, q2;

  // General values of v/u. Use a look-up table and series expansion.

  {
    double val = vbyu > 0.0625 ? vbyu : 0.063;
    int index = __spirv_ConvertFToS_Rint(__spirv_ocl_fma(256.0, val, 0.5));
    double2 tv = USE_TABLE(atan_jby256_tbl, index - 16);
    q1 = tv.s0;
    q2 = tv.s1;
    double c = (double)index * 0x1.0p-8;

    // We're going to scale u and v by 2^(-u_exponent) to bring them close to 1
    // u_exponent could be EMAX so we have to do it in 2 steps
    int m = -((int)(__clc_as_ulong(u) >> EXPSHIFTBITS_DP64) - EXPBIAS_DP64);
    // double um = __amdil_ldexp_f64(u, m);
    // double vm = __amdil_ldexp_f64(v, m);
    double um = __spirv_ocl_ldexp(u, m);
    double vm = __spirv_ocl_ldexp(v, m);

    // 26 leading bits of u
    double u1 = __clc_as_double(__clc_as_ulong(um) & 0xfffffffff8000000UL);
    double u2 = um - u1;

    double r = MATH_DIVIDE(__spirv_ocl_fma(-c, u2, __spirv_ocl_fma(-c, u1, vm)),
                           __spirv_ocl_fma(c, vm, um));

    // Polynomial approximation to atan(r)
    double s = r * r;
    q2 = q2 + __spirv_ocl_fma((s * __spirv_ocl_fma(-s, 0.19999918038989143496,
                                                   0.33333333333224095522)),
                              -r, r);
  }

  double q3, q4;
  {
    q3 = 0.0;
    q4 = vbyu;
  }

  double q5, q6;
  {
    double u1 = __clc_as_double(__clc_as_ulong(u) & 0xffffffff00000000UL);
    double u2 = u - u1;
    double vu1 = __clc_as_double(__clc_as_ulong(vbyu) & 0xffffffff00000000UL);
    double vu2 = vbyu - vu1;

    q5 = 0.0;
    double s = vbyu * vbyu;
    q6 = vbyu +
         __spirv_ocl_fma(
             -vbyu * s,
             __spirv_ocl_fma(
                 -s,
                 __spirv_ocl_fma(
                     -s,
                     __spirv_ocl_fma(-s,
                                     __spirv_ocl_fma(-s,
                                                     0.90029810285449784439E-01,
                                                     0.11110736283514525407),
                                     0.14285713561807169030),
                     0.19999999999393223405),
                 0.33333333333333170500),
             MATH_DIVIDE(
                 __spirv_ocl_fma(
                     -u, vu2,
                     __spirv_ocl_fma(-u2, vu1, __spirv_ocl_fma(-u1, vu1, v))),
                 u));
  }

  q3 = vbyu < 0x1.d12ed0af1a27fp-27 ? q3 : q5;
  q4 = vbyu < 0x1.d12ed0af1a27fp-27 ? q4 : q6;

  q1 = vbyu > 0.0625 ? q1 : q3;
  q2 = vbyu > 0.0625 ? q2 : q4;

  // Tidy-up according to which quadrant the arguments lie in
  double res1, res2, res3, res4;
  q1 = swap_vu ? piby2_head - q1 : q1;
  q2 = swap_vu ? piby2_tail - q2 : q2;
  q1 = xneg ? pi_head - q1 : q1;
  q2 = xneg ? pi_tail - q2 : q2;
  q1 = q1 + q2;
  res4 = yneg ? -q1 : q1;

  res1 = yneg ? -three_piby4 : three_piby4;
  res2 = yneg ? -piby4 : piby4;
  res3 = xneg ? res1 : res2;

  res3 = __spirv_IsInf(x2) && __spirv_IsInf(y2) ? res3 : res4;
  res1 = yneg ? -pi : pi;

  // abs(x)/abs(y) > 2^56 and x < 0
  res3 = (diffexp < -56 && xneg) ? res1 : res3;

  res4 = MATH_DIVIDE(y, x);
  // x positive and dominant over y by a factor of 2^28
  res3 = diffexp < -28 && xneg == 0 ? res4 : res3;

  // abs(y)/abs(x) > 2^56
  res4 = yneg ? -piby2 : piby2; // atan(y/x) is insignificant compared to piby2
  res3 = diffexp > 56 ? res4 : res3;

  res3 = x2 == 0.0 ? res4 : res3; // Zero x gives +- pi/2 depending on sign of y
  res4 = xneg ? res1 : y2;

  res3 = y2 == 0.0
             ? res4
             : res3; // Zero y gives +-0 for positive x and +-pi for negative x
  res3 = __spirv_IsNan(y2) ? y2 : res3;
  res3 = __spirv_IsNan(x2) ? x2 : res3;

  return res3;
}

_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_atan2, double,
                      double);

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_BINARY_BUILTIN(half, __spirv_ocl_atan2, __builtin_atan2f16, half, half)

#endif
