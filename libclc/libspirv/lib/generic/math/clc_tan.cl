/*
 * Copyright (c) 2014 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include "sincos_helpers.h"
#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/math/clc_fabs.h>
#include <clc/math/math.h>
#include <clc/math/tables.h>
#include <clc/relational/clc_isinf.h>
#include <clc/relational/clc_isnan.h>

#include <clc/math/tables.h>

_CLC_DEF _CLC_OVERLOAD float __clc_tan(float x) {
  int ix = __clc_as_int(x);
  int ax = ix & 0x7fffffff;
  float dx = __clc_as_float(ax);

  float r0, r1;
  int regn = __clc_argReductionS(&r0, &r1, dx);

  float t = __clc_tanf_piby4(r0 + r1, regn);
  t = __clc_as_float(__clc_as_int(t) ^ (ix ^ ax));

  t = ax >= PINFBITPATT_SP32 ? __clc_as_float(QNANBITPATT_SP32) : t;
  // Take care of subnormals
  t = (x == 0.0f) ? x : t;
  return t;
}
_CLC_UNARY_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, float, __clc_tan, float);

#ifdef cl_khr_fp64
#include "sincosD_piby4.h"

_CLC_DEF _CLC_OVERLOAD double __clc_tan(double x) {
  double y = __clc_fabs(x);

  double r, rr;
  int regn;

  if (y < 0x1.0p+30)
    __clc_remainder_piby2_medium(y, &r, &rr, &regn);
  else
    __clc_remainder_piby2_large(y, &r, &rr, &regn);

  double2 tt = __clc_tan_piby4(r, rr);

  int2 t = __clc_as_int2(regn & 1 ? tt.y : tt.x);
  t.hi ^= (x < 0.0) << 31;

  return __clc_isnan(x) || __clc_isinf(x) ? __clc_as_double(QNANBITPATT_DP64)
                                          : __clc_as_double(t);
}
_CLC_UNARY_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, double, __clc_tan, double);

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD half __clc_tan(half x) { return __clc_tan((float)x); }

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __clc_tan, half)

#endif
