//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef cl_khr_fp64

#include <libspirv/spirv.h>

#include "ep_log.h"
#include <clc/math/tables.h>
#include <clc/math/math.h>

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define LN0 8.33333333333317923934e-02
#define LN1 1.25000000037717509602e-02
#define LN2 2.23213998791944806202e-03
#define LN3 4.34887777707614552256e-04

#define LF0 8.33333333333333593622e-02
#define LF1 1.24999999978138668903e-02
#define LF2 2.23219810758559851206e-03

_CLC_DEF void __clc_ep_log(double x, int *xexp, double *r1, double *r2) {
  // Computes natural log(x). Algorithm based on:
  // Ping-Tak Peter Tang
  // "Table-driven implementation of the logarithm function in IEEE
  // floating-point arithmetic"
  // ACM Transactions on Mathematical Software (TOMS)
  // Volume 16, Issue 4 (December 1990)
  int near_one = x >= 0x1.e0faap-1 & x <= 0x1.1082cp+0;

  ulong ux = __clc_as_ulong(x);
  ulong uxs = __clc_as_ulong(__clc_as_double(0x03d0000000000000UL | ux) - 0x1.0p-962);
  int c = ux < IMPBIT_DP64;
  ux = c ? uxs : ux;
  int expadjust = c ? 60 : 0;

  // Store the exponent of x in xexp and put f into the range [0.5,1)
  int xexp1 = ((__clc_as_int2(ux).hi >> 20) & 0x7ff) - EXPBIAS_DP64 - expadjust;
  double f = __clc_as_double(HALFEXPBITS_DP64 | (ux & MANTBITS_DP64));
  *xexp = near_one ? 0 : xexp1;

  double r = x - 1.0;
  double u1 = MATH_DIVIDE(r, 2.0 + r);
  double ru1 = -r * u1;
  u1 = u1 + u1;

  int index = __clc_as_int2(ux).hi >> 13;
  index = ((0x80 | (index & 0x7e)) >> 1) + (index & 0x1);

  double f1 = index * 0x1.0p-7;
  double f2 = f - f1;
  double u2 = MATH_DIVIDE(f2, __spirv_ocl_fma(0.5, f2, f1));

  double z1 = USE_TABLE(ln_tbl_lo, (index - 64));;
  double q = USE_TABLE(ln_tbl_hi, (index - 64));

  z1 = near_one ? r : z1;
  q = near_one ? 0.0 : q;
  double u = near_one ? u1 : u2;
  double v = u * u;

  double cc = near_one ? ru1 : u2;

  double z21 = __spirv_ocl_fma(
      v, __spirv_ocl_fma(v, __spirv_ocl_fma(v, LN3, LN2), LN1), LN0);
  double z22 = __spirv_ocl_fma(v, __spirv_ocl_fma(v, LF2, LF1), LF0);
  double z2 = near_one ? z21 : z22;
  z2 = __spirv_ocl_fma(u * v, z2, cc) + q;

  *r1 = z1;
  *r2 = z2;
}

#endif
