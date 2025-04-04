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

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_tanh(float x) {
  // The definition of tanh(x) is sinh(x)/cosh(x), which is also equivalent
  // to the following three formulae:
  // 1.  (exp(x) - exp(-x))/(exp(x) + exp(-x))
  // 2.  (1 - (2/(exp(2*x) + 1 )))
  // 3.  (exp(2*x) - 1)/(exp(2*x) + 1)
  // but computationally, some formulae are better on some ranges.

  const float large_threshold = 0x1.0a2b24p+3f;

  uint ux = __clc_as_uint(x);
  uint aux = ux & EXSIGNBIT_SP32;
  uint xs = ux ^ aux;

  float y = __clc_as_float(aux);
  float y2 = y * y;

  float a1 = __spirv_ocl_mad(
      y2,
      __spirv_ocl_mad(y2, 0.4891631088530669873e-4F, -0.14628356048797849e-2F),
      -0.28192806108402678e0F);
  float b1 =
      __spirv_ocl_mad(y2, 0.3427017942262751343e0F, 0.845784192581041099e0F);

  float a2 = __spirv_ocl_mad(
      y2,
      __spirv_ocl_mad(y2, 0.3827534993599483396e-4F, -0.12325644183611929e-2F),
      -0.24069858695196524e0F);
  float b2 =
      __spirv_ocl_mad(y2, 0.292529068698052819e0F, 0.72209738473684982e0F);

  int c = y < 0.9f;
  float a = c ? a1 : a2;
  float b = c ? b1 : b2;
  float zlo = __spirv_ocl_mad(MATH_DIVIDE(a, b), y * y2, y);

  float p = __spirv_ocl_exp(2.0f * y) + 1.0f;
  float zhi = 1.0F - MATH_DIVIDE(2.0F, p);

  float z = y <= 1.0f ? zlo : zhi;
  z = __clc_as_float(xs | __clc_as_uint(z));

  // Edge cases
  float sone = __clc_as_float(0x3f800000U | xs);
  z = y > large_threshold ? sone : z;
  z = aux < 0x39000000 || aux > 0x7f800000 ? x : z;

  return z;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_tanh, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_tanh(double x) {
  // The definition of tanh(x) is sinh(x)/cosh(x), which is also equivalent
  // to the following three formulae:
  // 1.  (exp(x) - exp(-x))/(exp(x) + exp(-x))
  // 2.  (1 - (2/(exp(2*x) + 1 )))
  // 3.  (exp(2*x) - 1)/(exp(2*x) + 1)
  // but computationally, some formulae are better on some ranges.

  // The point at which e^-x is insignificant compared to e^x = ln(2^27)
  const double large_threshold = 0x1.2b708872320e2p+4;

  ulong ux = __clc_as_ulong(x);
  ulong ax = ux & ~SIGNBIT_DP64;
  ulong sx = ux ^ ax;
  double y = __clc_as_double(ax);
  double y2 = y * y;

  // y < 0.9
  double znl = __spirv_ocl_fma(
      y2,
      __spirv_ocl_fma(y2,
                      __spirv_ocl_fma(y2, -0.142077926378834722618091e-7,
                                      -0.200047621071909498730453e-3),
                      -0.176016349003044679402273e-1),
      -0.274030424656179760118928e0);

  double zdl = __spirv_ocl_fma(
      y2,
      __spirv_ocl_fma(y2,
                      __spirv_ocl_fma(y2, 0.2091140262529164482568557e-3,
                                      0.201562166026937652780575e-1),
                      0.381641414288328849317962e0),
      0.822091273968539282568011e0);

  // 0.9 <= y <= 1
  double znm = __spirv_ocl_fma(
      y2,
      __spirv_ocl_fma(y2,
                      __spirv_ocl_fma(y2, -0.115475878996143396378318e-7,
                                      -0.165597043903549960486816e-3),
                      -0.146173047288731678404066e-1),
      -0.227793870659088295252442e0);

  double zdm = __spirv_ocl_fma(
      y2,
      __spirv_ocl_fma(y2,
                      __spirv_ocl_fma(y2, 0.173076050126225961768710e-3,
                                      0.167358775461896562588695e-1),
                      0.317204558977294374244770e0),
      0.683381611977295894959554e0);

  int c = y < 0.9;
  double zn = c ? znl : znm;
  double zd = c ? zdl : zdm;
  double z = y + y * y2 * MATH_DIVIDE(zn, zd);

  // y > 1
  double p = __spirv_ocl_exp(2.0 * y) + 1.0;
  double zg = 1.0 - 2.0 / p;

  z = y > 1.0 ? zg : z;

  // Other cases
  z = y < 0x1.0p-28 || ax > PINFBITPATT_DP64 ? x : z;

  z = y > large_threshold ? 1.0 : z;

  return __clc_as_double(sx | __clc_as_ulong(z));
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_tanh, double);

#endif // cl_khr_fp64

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD half __spirv_ocl_tanh(half x) {
  return __spirv_ocl_tanh((float)x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __spirv_ocl_tanh, half)

#endif
