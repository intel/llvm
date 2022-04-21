//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mangle_common.h"
#include <spirv/spirv.h>
#include <utils.h>

double __ocml_sincos_f64(double, double *);
float __ocml_sincos_f32(float, float *);

#define FUNCNAME(IN, OUT)                                                      \
  __CLC_XCONCAT(__CLC_XCONCAT(_Z18__spirv_ocl_sincos, IN), OUT)
#define VEC_TYPE(T, N) __CLC_XCONCAT(__CLC_XCONCAT(__CLC_XCONCAT(Dv, N), _), T)
#define VEC_FUNCNAME(N, MANGLED_TYPE, MANGLED_PTR)                             \
  FUNCNAME(VEC_TYPE(MANGLED_TYPE, N), __CLC_XCONCAT(MANGLED_PTR, S_))

#define MANUALLY_MANGLED_SINCOS_IMPL(ADDRSPACE, BUILTIN, ARG1_TYPE,            \
                                     MANGLED_ARG1_TYPE, MANGLED_POINTER_TYPE,  \
                                     FP_TYPE)                                  \
  _CLC_DEF ARG1_TYPE FUNCNAME(MANGLED_ARG1_TYPE, MANGLED_POINTER_TYPE)(        \
      ARG1_TYPE x, __attribute((address_space(ADDRSPACE))) ARG1_TYPE * ptr) {  \
    FP_TYPE cos_val;                                                           \
    FP_TYPE sin_val = BUILTIN(x, &cos_val);                                    \
    *ptr = cos_val;                                                            \
    return sin_val;                                                            \
  }

#define __CLC_SINCOS(BUILTIN, ARG_TYPE, MANGLED_ARG_TYPE, FP_TYPE)             \
  MANUALLY_MANGLED_SINCOS_IMPL(0, BUILTIN, ARG_TYPE, MANGLED_ARG_TYPE,         \
                               __CLC_XCONCAT(P, MANGLED_ARG_TYPE), FP_TYPE)    \
  MANUALLY_MANGLED_SINCOS_IMPL(1, BUILTIN, ARG_TYPE, MANGLED_ARG_TYPE,         \
                               __CLC_XCONCAT(PU3AS1, MANGLED_ARG_TYPE),        \
                               FP_TYPE)                                        \
  MANUALLY_MANGLED_SINCOS_IMPL(3, BUILTIN, ARG_TYPE, MANGLED_ARG_TYPE,         \
                               __CLC_XCONCAT(PU3AS3, MANGLED_ARG_TYPE),        \
                               FP_TYPE)                                        \
  MANUALLY_MANGLED_SINCOS_IMPL(5, BUILTIN, ARG_TYPE, MANGLED_ARG_TYPE,         \
                               __CLC_XCONCAT(PU3AS5, MANGLED_ARG_TYPE),        \
                               FP_TYPE)

#define FNAME_GENERIC(N) VEC_FUNCNAME(N, f, P)
#define FNAME_GLOBAL(N) VEC_FUNCNAME(N, f, PU3AS1)
#define FNAME_LOCAL(N) VEC_FUNCNAME(N, f, PU3AS3)
#define FNAME_PRIVATE(N) VEC_FUNCNAME(N, f, PU3AS5)

__CLC_SINCOS(__ocml_sincos_f32, float, f, float)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(f, Pf), FNAME_GENERIC, float, 0,
                                  float)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(f, PU3AS1f), FNAME_GLOBAL, float, 1,
                                  float)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(f, PU3AS3f), FNAME_LOCAL, float, 3,
                                  float)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(f, PU3AS5f), FNAME_PRIVATE, float, 5,
                                  float)

#undef FNAME_GENERIC
#undef FNAME_GLOBAL
#undef FNAME_LOCAL
#undef FNAME_PRIVATE

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define FNAME_GENERIC(N) VEC_FUNCNAME(N, d, P)
#define FNAME_GLOBAL(N) VEC_FUNCNAME(N, d, PU3AS1)
#define FNAME_LOCAL(N) VEC_FUNCNAME(N, d, PU3AS3)
#define FNAME_PRIVATE(N) VEC_FUNCNAME(N, d, PU3AS5)

__CLC_SINCOS(__ocml_sincos_f64, double, d, double)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(d, Pd), FNAME_GENERIC, double, 0,
                                  double)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(d, PU3AS1d), FNAME_GLOBAL, double, 1,
                                  double)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(d, PU3AS3d), FNAME_LOCAL, double, 3,
                                  double)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(d, PU3AS5d), FNAME_PRIVATE, double,
                                  5, double)

#undef FNAME_GENERIC
#undef FNAME_GLOBAL
#undef FNAME_LOCAL
#undef FNAME_PRIVATE

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define FNAME_GENERIC(N) VEC_FUNCNAME(N, Dh, P)
#define FNAME_GLOBAL(N) VEC_FUNCNAME(N, Dh, PU3AS1)
#define FNAME_LOCAL(N) VEC_FUNCNAME(N, Dh, PU3AS3)
#define FNAME_PRIVATE(N) VEC_FUNCNAME(N, Dh, PU3AS5)

__CLC_SINCOS(__ocml_sincos_f32, half, Dh, float)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(Dh, PDh), FNAME_GENERIC, half, 0,
                                  half)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(Dh, PU3AS1Dh), FNAME_GLOBAL, half, 1,
                                  half)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(Dh, PU3AS3Dh), FNAME_LOCAL, half, 3,
                                  half)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(Dh, PU3AS5Dh), FNAME_PRIVATE, half,
                                  5, half)

#undef FNAME_GENERIC
#undef FNAME_GLOBAL
#undef FNAME_LOCAL
#undef FNAME_PRIVATE

#endif
