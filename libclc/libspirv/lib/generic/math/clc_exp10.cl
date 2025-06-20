//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc_convert.h>
#include <clc/clcmacro.h>
#include <clc/math/clc_fma.h>
#include <clc/math/clc_mad.h>
#include <clc/math/clc_subnormal_config.h>
#include <clc/math/math.h>
#include <clc/math/tables.h>
#include <clc/opencl/clc.h>
#include <clc/relational/clc_isnan.h>
#include <libspirv/spirv.h>

//    Algorithm:
//
//    e^x = 2^(x/ln(2)) = 2^(x*(64/ln(2))/64)
//
//    x*(64/ln(2)) = n + f, |f| <= 0.5, n is integer
//    n = 64*m + j,   0 <= j < 64
//
//    e^x = 2^((64*m + j + f)/64)
//        = (2^m) * (2^(j/64)) * 2^(f/64)
//        = (2^m) * (2^(j/64)) * e^(f*(ln(2)/64))
//
//    f = x*(64/ln(2)) - n
//    r = f*(ln(2)/64) = x - n*(ln(2)/64)
//
//    e^x = (2^m) * (2^(j/64)) * e^r
//
//    (2^(j/64)) is precomputed
//
//    e^r = 1 + r + (r^2)/2! + (r^3)/3! + (r^4)/4! + (r^5)/5!
//    e^r = 1 + q
//
//    q = r + (r^2)/2! + (r^3)/3! + (r^4)/4! + (r^5)/5!
//
//    e^x = (2^m) * ( (2^(j/64)) + q*(2^(j/64)) )

_CLC_DEF _CLC_OVERLOAD float __clc_exp10(float x) {
  // 128*log2/log10 : 38.53183944498959
  const float X_MAX = 0x1.344134p+5f;
  // -149*log2/log10 : -44.8534693539332
  const float X_MIN = -0x1.66d3e8p+5f;
  // 64*log10/log2 : 212.6033980727912
  const float R_64_BY_LOG10_2 = 0x1.a934f0p+7f;
  // log2/(64 * log10) lead : 0.004699707
  const float R_LOG10_2_BY_64_LD = 0x1.340000p-8f;
  // log2/(64 * log10) tail : 0.00000388665057
  const float R_LOG10_2_BY_64_TL = 0x1.04d426p-18f;
  const float R_LN10 = 0x1.26bb1cp+1f;

  int return_nan = __clc_isnan(x);
  int return_inf = x > X_MAX;
  int return_zero = x < X_MIN;

  int n = __clc_convert_int(x * R_64_BY_LOG10_2);

  float fn = (float)n;
  int j = n & 0x3f;
  int m = n >> 6;
  int m2 = m << EXPSHIFTBITS_SP32;
  float r;

  r = R_LN10 *
      __clc_mad(fn, -R_LOG10_2_BY_64_TL, __clc_mad(fn, -R_LOG10_2_BY_64_LD, x));

  // Truncated Taylor series for e^r
  float z2 = __clc_mad(__clc_mad(__clc_mad(r, 0x1.555556p-5f, 0x1.555556p-3f),
                                 r, 0x1.000000p-1f),
                       r * r, r);

  float two_to_jby64 = USE_TABLE(exp_tbl, j);
  z2 = __clc_mad(two_to_jby64, z2, two_to_jby64);

  float z2s = z2 * __clc_as_float(0x1 << (m + 149));
  float z2n = __clc_as_float(__clc_as_int(z2) + m2);
  z2 = m <= -126 ? z2s : z2n;

  z2 = return_inf ? __clc_as_float(PINFBITPATT_SP32) : z2;
  z2 = return_zero ? 0.0f : z2;
  z2 = return_nan ? x : z2;
  return z2;
}
_CLC_UNARY_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, float, __clc_exp10, float)

#ifdef cl_khr_fp64
_CLC_DEF _CLC_OVERLOAD double __clc_exp10(double x) {
  // 1024*ln(2)/ln(10)
  const double X_MAX = 0x1.34413509f79ffp+8;
  // -1074*ln(2)/ln(10)
  const double X_MIN = -0x1.434e6420f4374p+8;
  // 64*ln(10)/ln(2)
  const double R_64_BY_LOG10_2 = 0x1.a934f0979a371p+7;
  // head ln(2)/(64*ln(10))
  const double R_LOG10_2_BY_64_LD = 0x1.3441350000000p-8;
  // tail ln(2)/(64*ln(10))
  const double R_LOG10_2_BY_64_TL = 0x1.3ef3fde623e25p-37;
  // ln(10)
  const double R_LN10 = 0x1.26bb1bbb55516p+1;

  int n = __clc_convert_int(x * R_64_BY_LOG10_2);

  double dn = (double)n;

  int j = n & 0x3f;
  int m = n >> 6;

  double r =
      R_LN10 * __spirv_ocl_fma(-R_LOG10_2_BY_64_TL, dn,
                               __spirv_ocl_fma(-R_LOG10_2_BY_64_LD, dn, x));

  // 6 term tail of Taylor expansion of e^r
  double z2 =
      r * __spirv_ocl_fma(
              r,
              __spirv_ocl_fma(
                  r,
                  __spirv_ocl_fma(
                      r,
                      __spirv_ocl_fma(r,
                                      __spirv_ocl_fma(r, 0x1.6c16c16c16c17p-10,
                                                      0x1.1111111111111p-7),
                                      0x1.5555555555555p-5),
                      0x1.5555555555555p-3),
                  0x1.0000000000000p-1),
              1.0);

  double2 tv;
  tv.s0 = USE_TABLE(two_to_jby64_ep_tbl_head, j);
  tv.s1 = USE_TABLE(two_to_jby64_ep_tbl_tail, j);
  z2 = __spirv_ocl_fma(tv.s0 + tv.s1, z2, tv.s1) + tv.s0;

  int small_value = (m < -1022) || ((m == -1022) && (z2 < 1.0));

  int n1 = m >> 2;
  int n2 = m - n1;
  double z3 = z2 * __clc_as_double(((long)n1 + 1023) << 52);
  z3 *= __clc_as_double(((long)n2 + 1023) << 52);

  z2 = __spirv_ocl_ldexp(z2, m);
  z2 = small_value ? z3 : z2;

  z2 = __clc_isnan(x) ? x : z2;

  z2 = x > X_MAX ? __clc_as_double(PINFBITPATT_DP64) : z2;
  z2 = x < X_MIN ? 0.0 : z2;

  return z2;
}
_CLC_UNARY_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, double, __clc_exp10, double)
#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_UNARY_BUILTIN_SCALARIZE(half, __clc_exp10, __builtin_exp10f16, half)

#endif
