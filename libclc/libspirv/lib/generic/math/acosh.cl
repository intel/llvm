//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#include "ep_log.h"
#include <clc/clcmacro.h>
#include <clc/math/math.h>

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_acosh(float x) {
  uint ux = __clc_as_uint(x);

  // Arguments greater than 1/sqrt(epsilon) in magnitude are
  // approximated by acosh(x) = ln(2) + ln(x)
  // For 2.0 <= x <= 1/sqrt(epsilon) the approximation is
  // acosh(x) = ln(x + sqrt(x*x-1)) */
  int high = ux > 0x46000000U;
  int med = ux > 0x40000000U;

  float w = x - 1.0f;
  float s = w * w + 2.0f * w;
  float t = x * x - 1.0f;
  float r = __spirv_ocl_sqrt(med ? t : s) + (med ? x : w);
  float v = (high ? x : r) - (med ? 1.0f : 0.0f);
  float z = __spirv_ocl_log1p(v) + (high ? 0x1.62e430p-1f : 0.0f);

  z = ux >= PINFBITPATT_SP32 ? x : z;
  z = x < 1.0f ? __clc_as_float(QNANBITPATT_SP32) : z;

  return z;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_acosh, float)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_acosh(double x) {
  const double recrteps =
      0x1.6a09e667f3bcdp+26; // 1/sqrt(eps) = 9.49062656242515593767e+07
  // log2_lead and log2_tail sum to an extra-precise version of log(2)
  const double log2_lead = 0x1.62e42ep-1;
  const double log2_tail = 0x1.efa39ef35793cp-25;

  // Handle x >= 128 here
  int xlarge = x > recrteps;
  double r = x + __spirv_ocl_sqrt(__spirv_ocl_fma(x, x, -1.0));
  r = xlarge ? x : r;

  int xexp;
  double r1, r2;
  __clc_ep_log(r, &xexp, &r1, &r2);

  double dxexp = xexp + xlarge;
  r1 = __spirv_ocl_fma(dxexp, log2_lead, r1);
  r2 = __spirv_ocl_fma(dxexp, log2_tail, r2);

  double ret1 = r1 + r2;

  // Handle 1 < x < 128 here
  // We compute the value
  // t = x - 1.0 + sqrt(2.0*(x - 1.0) + (x - 1.0)*(x - 1.0))
  // using simulated quad precision.
  double t = x - 1.0;
  double u1 = t * 2.0;

  // (t,0) * (t,0) -> (v1, v2)
  double v1 = t * t;
  double v2 = __spirv_ocl_fma(t, t, -v1);

  // (u1,0) + (v1,v2) -> (w1,w2)
  r = u1 + v1;
  double s = (((u1 - r) + v1) + v2);
  double w1 = r + s;
  double w2 = (r - w1) + s;

  // sqrt(w1,w2) -> (u1,u2)
  double p1 = __spirv_ocl_sqrt(w1);
  double a1 = p1 * p1;
  double a2 = __spirv_ocl_fma(p1, p1, -a1);
  double temp = (((w1 - a1) - a2) + w2);
  double p2 = MATH_DIVIDE(temp * 0.5, p1);
  u1 = p1 + p2;
  double u2 = (p1 - u1) + p2;

  // (u1,u2) + (t,0) -> (r1,r2)
  r = u1 + t;
  s = ((u1 - r) + t) + u2;
  // r1 = r + s;
  // r2 = (r - r1) + s;
  // t = r1 + r2;
  t = r + s;

  // For arguments 1.13 <= x <= 1.5 the log1p function is good enough
  double ret2 = __spirv_ocl_log1p(t);

  ulong ux = __clc_as_ulong(x);
  double ret = x >= 128.0 ? ret1 : ret2;

  ret = ux >= 0x7FF0000000000000 ? x : ret;
  ret = x == 1.0 ? 0.0 : ret;
  ret =
      (ux & SIGNBIT_DP64) != 0UL || x < 1.0 ? __clc_as_double(QNANBITPATT_DP64) : ret;

  return ret;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_acosh, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_UNARY_BUILTIN_SCALARIZE(half, __spirv_ocl_acosh, __builtin_acoshf, half)

#endif
