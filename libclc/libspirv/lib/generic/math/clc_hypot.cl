//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/integer/clc_abs.h>
#include <clc/math/clc_mad.h>
#include <clc/math/clc_subnormal_config.h>
#include <clc/math/math.h>
#include <clc/relational/clc_isnan.h>
#include <clc/shared/clc_clamp.h>
#include <math/clc_hypot.h>

// Returns sqrt(x*x + y*y) with no overflow or underflow unless the result
// warrants it
_CLC_DEF _CLC_OVERLOAD float __clc_hypot(float x, float y) {
  uint ux = __clc_as_uint(x);
  uint aux = ux & EXSIGNBIT_SP32;
  uint uy = __clc_as_uint(y);
  uint auy = uy & EXSIGNBIT_SP32;
  float retval;
  int c = aux > auy;
  ux = c ? aux : auy;
  uy = c ? auy : aux;

  int xexp =
      __clc_clamp((int)(ux >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32, -126, 126);
  float fx_exp = __clc_as_float((xexp + EXPBIAS_SP32) << EXPSHIFTBITS_SP32);
  float fi_exp = __clc_as_float((-xexp + EXPBIAS_SP32) << EXPSHIFTBITS_SP32);
  float fx = __clc_as_float(ux) * fi_exp;
  float fy = __clc_as_float(uy) * fi_exp;
  retval = __spirv_ocl_sqrt(__clc_mad(fx, fx, fy * fy)) * fx_exp;

  retval = ux > PINFBITPATT_SP32 || uy == 0 ? __clc_as_float(ux) : retval;
  retval = ux == PINFBITPATT_SP32 || uy == PINFBITPATT_SP32
               ? __clc_as_float(PINFBITPATT_SP32)
               : retval;
  return retval;
}
_CLC_BINARY_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, float, __clc_hypot, float, float)

#ifdef cl_khr_fp64
_CLC_DEF _CLC_OVERLOAD double __clc_hypot(double x, double y) {
  ulong ux = __clc_as_ulong(x) & ~SIGNBIT_DP64;
  int xexp = ux >> EXPSHIFTBITS_DP64;
  x = __clc_as_double(ux);

  ulong uy = __clc_as_ulong(y) & ~SIGNBIT_DP64;
  int yexp = uy >> EXPSHIFTBITS_DP64;
  y = __clc_as_double(uy);

  int c = xexp > EXPBIAS_DP64 + 500 | yexp > EXPBIAS_DP64 + 500;
  double preadjust = c ? 0x1.0p-600 : 1.0;
  double postadjust = c ? 0x1.0p+600 : 1.0;

  c = xexp < EXPBIAS_DP64 - 500 | yexp < EXPBIAS_DP64 - 500;
  preadjust = c ? 0x1.0p+600 : preadjust;
  postadjust = c ? 0x1.0p-600 : postadjust;

  double ax = x * preadjust;
  double ay = y * preadjust;

  // The post adjust may overflow, but this can't be avoided in any case
  double r = __spirv_ocl_sqrt(__spirv_ocl_fma(ax, ax, ay * ay)) * postadjust;

  // If the difference in exponents between x and y is large
  double s = x + y;
  c = __clc_abs(xexp - yexp) > MANTLENGTH_DP64 + 1;
  r = c ? s : r;

  // Check for NaN
  // c = x != x | y != y;
  c = __clc_isnan(x) | __clc_isnan(y);
  r = c ? __clc_as_double(QNANBITPATT_DP64) : r;

  // If either is Inf, we must return Inf
  c = x == __clc_as_double(PINFBITPATT_DP64) | y == __clc_as_double(PINFBITPATT_DP64);
  r = c ? __clc_as_double(PINFBITPATT_DP64) : r;

  return r;
}

_CLC_BINARY_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, double, __clc_hypot, double,
                      double)
#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_BINARY_BUILTIN(half, __clc_hypot, __builtin_hypotf, half, half)

#endif
