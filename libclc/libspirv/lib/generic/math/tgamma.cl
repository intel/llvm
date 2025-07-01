//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#include <clc/math/math.h>

_CLC_OVERLOAD _CLC_DEF float __spirv_ocl_tgamma(float x) {
  const float pi = 3.1415926535897932384626433832795f;
  float ax = __spirv_ocl_fabs(x);
  float lg = __spirv_ocl_lgamma(ax);
  float g = __spirv_ocl_exp(lg);

  if (x < 0.0f) {
    float z = __spirv_ocl_sinpi(x);
    g = g * ax * z;
    g = pi / g;
    g = g == 0 ? __clc_as_float(PINFBITPATT_SP32) : g;
    g = z == 0 ? __clc_as_float(QNANBITPATT_SP32) : g;
  }

  return g;
}

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __spirv_ocl_tgamma(double x) {
  const double pi = 3.1415926535897932384626433832795;
  double ax = __spirv_ocl_fabs(x);
  double lg = __spirv_ocl_lgamma(ax);
  double g = __spirv_ocl_exp(lg);

  if (x < 0.0) {
    double z = __spirv_ocl_sinpi(x);
    g = g * ax * z;
    g = pi / g;
    g = g == 0 ? __clc_as_double(PINFBITPATT_DP64) : g;
    g = z == 0 ? __clc_as_double(QNANBITPATT_DP64) : g;
  }

  return g;
}

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD half __spirv_ocl_tgamma(half x) {
  return __spirv_ocl_tgamma((float)x);
}

#endif

#define FUNCTION __spirv_ocl_tgamma
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
