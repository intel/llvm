//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/clc_subnormal_config.h>
#include <clc/math/math.h>
#include <libspirv/spirv.h>

float __ocml_ldexp_f32(float, int);
_CLC_OVERLOAD _CLC_INLINE static float ocml_ldexp_helper(float x, int y) {
  return __ocml_ldexp_f32(x, y);
}

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
double __ocml_ldexp_f64(double, int);
_CLC_OVERLOAD _CLC_INLINE static double ocml_ldexp_helper(double x, int y) {
  return __ocml_ldexp_f64(x, y);
}
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
half __ocml_ldexp_f16(half, int);
_CLC_OVERLOAD _CLC_INLINE static half ocml_ldexp_helper(half x, int y) {
  return __ocml_ldexp_f16(x, y);
}
#endif

#define __CLC_FUNCTION __spirv_ocl_ldexp
#define __CLC_IMPL_FUNCTION ocml_ldexp_helper
#undef __CLC_MIN_VECSIZE
#define __CLC_MIN_VECSIZE 1

#define __CLC_ARG2_TYPE int
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_ARG2_TYPE

#define __CLC_ARG2_TYPE uint
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_ARG2_TYPE
