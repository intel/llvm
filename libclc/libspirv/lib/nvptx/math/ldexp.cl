//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#include <clc/math/clc_subnormal_config.h>
#include <clc/math/math.h>
#include <libspirv/ptx-nvidiacl/libdevice.h>

_CLC_OVERLOAD _CLC_INLINE static float nv_ldexp_helper(float x, int y) {
  return __nv_ldexpf(x, y);
}

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
_CLC_OVERLOAD _CLC_INLINE static double nv_ldexp_helper(double x, int y) {
  return __nv_ldexp(x, y);
}
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
_CLC_OVERLOAD _CLC_INLINE static half nv_ldexp_helper(half x, int y) {
  return __nv_ldexpf(x, y);
}
#endif

#define __CLC_FUNCTION __spirv_ocl_ldexp
#define __CLC_IMPL_FUNCTION nv_ldexp_helper
#define __CLC_MIN_VECSIZE 1

#define __CLC_ARG2_TYPE int
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_ARG2_TYPE

#define __CLC_ARG2_TYPE uint
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_ARG2_TYPE
