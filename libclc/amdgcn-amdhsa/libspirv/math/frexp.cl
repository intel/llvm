//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include <utils.h>

double __ocml_frexp_f64(double, int *);
float __ocml_frexp_f32(float, int *);

_CLC_OVERLOAD _CLC_DEF float __clc_spirv_ocl_frexp(float x, private int *ep) {
  return __ocml_frexp_f32(x, ep);
}

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __clc_spirv_ocl_frexp(double x, private int *ep) {
  return __ocml_frexp_f64(x, ep);
}

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half __clc_spirv_ocl_frexp(half x, private int *ep) {
  float t = x;
  return __ocml_frexp_f32(t, ep);
}

#endif

#define __CLC_ADDRESS_SPACE private
#define __CLC_GENTYPE float
#include <frexp.inc>
#undef __CLC_GENTYPE
#ifdef cl_khr_fp64
#define __CLC_GENTYPE double
#include <frexp.inc>
#undef __CLC_GENTYPE
#endif
#ifdef cl_khr_fp16
#define __CLC_GENTYPE half
#include <frexp.inc>
#undef __CLC_GENTYPE
#endif
#undef __CLC_ADDRESS_SPACE

#define __CLC_ADDRESS_SPACE global
#define __CLC_GENTYPE float
#include <frexp.inc>
#undef __CLC_GENTYPE
#ifdef cl_khr_fp64
#define __CLC_GENTYPE double
#include <frexp.inc>
#undef __CLC_GENTYPE
#endif
#ifdef cl_khr_fp16
#define __CLC_GENTYPE half
#include <frexp.inc>
#undef __CLC_GENTYPE
#endif
#undef __CLC_ADDRESS_SPACE

#define __CLC_ADDRESS_SPACE local
#define __CLC_GENTYPE float
#include <frexp.inc>
#undef __CLC_GENTYPE
#ifdef cl_khr_fp64
#define __CLC_GENTYPE double
#include <frexp.inc>
#undef __CLC_GENTYPE
#endif
#ifdef cl_khr_fp16
#define __CLC_GENTYPE half
#include <frexp.inc>
#undef __CLC_GENTYPE
#endif
#undef __CLC_ADDRESS_SPACE
