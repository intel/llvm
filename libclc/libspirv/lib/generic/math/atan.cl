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

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_atan(float x) {
  const float piby2 = 1.5707963267948966f; // 0x3ff921fb54442d18

  uint ux = __clc_as_uint(x);
  uint aux = ux & EXSIGNBIT_SP32;
  uint sx = ux ^ aux;

  float spiby2 = __clc_as_float(sx | __clc_as_uint(piby2));

  float v = __clc_as_float(aux);

  // Return for NaN
  float ret = x;

  // 2^26 <= |x| <= Inf => atan(x) is close to piby2
  ret = aux <= PINFBITPATT_SP32 ? spiby2 : ret;

  // Reduce arguments 2^-19 <= |x| < 2^26

  // 39/16 <= x < 2^26
  x = -MATH_RECIP(v);
  float c = 1.57079632679489655800f; // atan(infinity)

  // 19/16 <= x < 39/16
  int l = aux < 0x401c0000;
  float xx = MATH_DIVIDE(v - 1.5f, __spirv_ocl_mad(v, 1.5f, 1.0f));
  x = l ? xx : x;
  c = l ? 9.82793723247329054082e-01f : c; // atan(1.5)

  // 11/16 <= x < 19/16
  l = aux < 0x3f980000U;
  xx = MATH_DIVIDE(v - 1.0f, 1.0f + v);
  x = l ? xx : x;
  c = l ? 7.85398163397448278999e-01f : c; // atan(1)

  // 7/16 <= x < 11/16
  l = aux < 0x3f300000;
  xx = MATH_DIVIDE(__spirv_ocl_mad(v, 2.0f, -1.0f), 2.0f + v);
  x = l ? xx : x;
  c = l ? 4.63647609000806093515e-01f : c; // atan(0.5)

  // 2^-19 <= x < 7/16
  l = aux < 0x3ee00000;
  x = l ? v : x;
  c = l ? 0.0f : c;

  // Core approximation: Remez(2,2) on [-7/16,7/16]

  float s = x * x;
  float a =
      __spirv_ocl_mad(s,
                      __spirv_ocl_mad(s, 0.470677934286149214138357545549e-2f,
                                      0.192324546402108583211697690500f),
                      0.296528598819239217902158651186f);

  float b =
      __spirv_ocl_mad(s,
                      __spirv_ocl_mad(s, 0.299309699959659728404442796915f,
                                      0.111072499995399550138837673349e1f),
                      0.889585796862432286486651434570f);

  float q = x * s * MATH_DIVIDE(a, b);

  float z = c - (q - x);
  float zs = __clc_as_float(sx | __clc_as_uint(z));

  ret = aux < 0x4c800000 ? zs : ret;

  // |x| < 2^-19
  ret = aux < 0x36000000 ? __clc_as_float(ux) : ret;
  return ret;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_atan, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_atan(double x) {
  const double piby2 = 1.5707963267948966e+00; // 0x3ff921fb54442d18

  double v = __spirv_ocl_fabs(x);

  // 2^56 > v > 39/16
  double a = -1.0;
  double b = v;
  // (chi + clo) = arctan(infinity)
  double chi = 1.57079632679489655800e+00;
  double clo = 6.12323399573676480327e-17;

  double ta = v - 1.5;
  double tb = 1.0 + 1.5 * v;
  int l = v <= 0x1.38p+1; // 39/16 > v > 19/16
  a = l ? ta : a;
  b = l ? tb : b;
  // (chi + clo) = arctan(1.5)
  chi = l ? 9.82793723247329054082e-01 : chi;
  clo = l ? 1.39033110312309953701e-17 : clo;

  ta = v - 1.0;
  tb = 1.0 + v;
  l = v <= 0x1.3p+0; // 19/16 > v > 11/16
  a = l ? ta : a;
  b = l ? tb : b;
  // (chi + clo) = arctan(1.)
  chi = l ? 7.85398163397448278999e-01 : chi;
  clo = l ? 3.06161699786838240164e-17 : clo;

  ta = 2.0 * v - 1.0;
  tb = 2.0 + v;
  l = v <= 0x1.6p-1; // 11/16 > v > 7/16
  a = l ? ta : a;
  b = l ? tb : b;
  // (chi + clo) = arctan(0.5)
  chi = l ? 4.63647609000806093515e-01 : chi;
  clo = l ? 2.26987774529616809294e-17 : clo;

  l = v <= 0x1.cp-2; // v < 7/16
  a = l ? v : a;
  b = l ? 1.0 : b;
  ;
  chi = l ? 0.0 : chi;
  clo = l ? 0.0 : clo;

  // Core approximation: Remez(4,4) on [-7/16,7/16]
  double r = a / b;
  double s = r * r;
  double qn = __spirv_ocl_fma(
      s,
      __spirv_ocl_fma(
          s,
          __spirv_ocl_fma(s,
                          __spirv_ocl_fma(s, 0.142316903342317766e-3,
                                          0.304455919504853031e-1),
                          0.220638780716667420e0),
          0.447677206805497472e0),
      0.268297920532545909e0);

  double qd = __spirv_ocl_fma(
      s,
      __spirv_ocl_fma(
          s,
          __spirv_ocl_fma(s,
                          __spirv_ocl_fma(s, 0.389525873944742195e-1,
                                          0.424602594203847109e0),
                          0.141254259931958921e1),
          0.182596787737507063e1),
      0.804893761597637733e0);

  double q = r * s * qn / qd;
  r = chi - ((q - clo) - r);

  double z = __spirv_IsNan(x) ? x : piby2;
  z = v <= 0x1.0p+56 ? r : z;
  z = v < 0x1.0p-26 ? v : z;
  return x == v ? z : -z;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_atan, double);

#endif // cl_khr_fp64

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_UNARY_BUILTIN_SCALARIZE(half, __spirv_ocl_atan, __builtin_atanf16, half)

#endif
