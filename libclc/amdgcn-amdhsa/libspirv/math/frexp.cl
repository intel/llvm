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

double __ocml_frexp_f64(double, int *);
float __ocml_frexp_f32(float, int *);

#define FUNCNAME(IN, OUT)                                                      \
  __CLC_XCONCAT(__CLC_XCONCAT(_Z17__spirv_ocl_frexp, IN), OUT)
#define VEC_TYPE(T, N) __CLC_XCONCAT(__CLC_XCONCAT(__CLC_XCONCAT(Dv, N), _), T)
#define VEC_FUNCNAME(N, MANGLED_IN_TYPE, MANGLED_PTR, MANGLED_OUT_TYPE)        \
  FUNCNAME(VEC_TYPE(MANGLED_IN_TYPE, N),                                       \
           __CLC_XCONCAT(MANGLED_PTR, VEC_TYPE(MANGLED_OUT_TYPE, N)))

#define MANUALLY_MANGLED_FREXP_IMPL(ADDRSPACE, BUILTIN, ARG1_TYPE,             \
                                    MANGLED_ARG1_TYPE, MANGLED_ARG2_TYPE)      \
  _CLC_DEF ARG1_TYPE FUNCNAME(MANGLED_ARG1_TYPE, MANGLED_ARG2_TYPE)(           \
      ARG1_TYPE x, __attribute((address_space(ADDRSPACE))) int *ptr) {         \
    int stack_iptr;                                                            \
    ARG1_TYPE ret = BUILTIN(x, &stack_iptr);                                   \
    *ptr = stack_iptr;                                                         \
    return ret;                                                                \
  }

#define __CLC_FREXP(BUILTIN, ARG_TYPE, MANGLED_ARG1_TYPE)                      \
  MANUALLY_MANGLED_FREXP_IMPL(0, BUILTIN, ARG_TYPE, MANGLED_ARG1_TYPE, Pi)     \
  MANUALLY_MANGLED_FREXP_IMPL(1, BUILTIN, ARG_TYPE, MANGLED_ARG1_TYPE,         \
                              PU3AS1i)                                         \
  MANUALLY_MANGLED_FREXP_IMPL(3, BUILTIN, ARG_TYPE, MANGLED_ARG1_TYPE,         \
                              PU3AS3i)                                         \
  MANUALLY_MANGLED_FREXP_IMPL(5, BUILTIN, ARG_TYPE, MANGLED_ARG1_TYPE, PU3AS5i)

#define FNAME_GENERIC(N) VEC_FUNCNAME(N, f, P, i)
#define FNAME_GLOBAL(N) VEC_FUNCNAME(N, f, PU3AS1, i)
#define FNAME_LOCAL(N) VEC_FUNCNAME(N, f, PU3AS3, i)
#define FNAME_PRIVATE(N) VEC_FUNCNAME(N, f, PU3AS5, i)

__CLC_FREXP(__ocml_frexp_f32, float, f)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(f, Pi), FNAME_GENERIC, float, 0, int)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(f, PU3AS1i), FNAME_GLOBAL, float, 1,
                                  int)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(f, PU3AS3i), FNAME_LOCAL, float, 3,
                                  int)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(f, PU3AS5i), FNAME_PRIVATE, float, 5,
                                  int)

#undef FNAME_GENERIC
#undef FNAME_GLOBAL
#undef FNAME_LOCAL
#undef FNAME_PRIVATE

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define FNAME_GENERIC(N) VEC_FUNCNAME(N, d, P, i)
#define FNAME_GLOBAL(N) VEC_FUNCNAME(N, d, PU3AS1, i)
#define FNAME_LOCAL(N) VEC_FUNCNAME(N, d, PU3AS3, i)
#define FNAME_PRIVATE(N) VEC_FUNCNAME(N, d, PU3AS5, i)

__CLC_FREXP(__ocml_frexp_f64, double, d)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(d, Pi), FNAME_GENERIC, double, 0,
                                  int)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(d, PU3AS1i), FNAME_GLOBAL, double, 1,
                                  int)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(d, PU3AS3i), FNAME_LOCAL, double, 3,
                                  int)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(d, PU3AS5i), FNAME_PRIVATE, double,
                                  5, int)

#undef FNAME_GENERIC
#undef FNAME_GLOBAL
#undef FNAME_LOCAL
#undef FNAME_PRIVATE

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define FNAME_GENERIC(N) VEC_FUNCNAME(N, Dh, P, i)
#define FNAME_GLOBAL(N) VEC_FUNCNAME(N, Dh, PU3AS1, i)
#define FNAME_LOCAL(N) VEC_FUNCNAME(N, Dh, PU3AS3, i)
#define FNAME_PRIVATE(N) VEC_FUNCNAME(N, Dh, PU3AS5, i)

__CLC_FREXP(__ocml_frexp_f32, half, Dh)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(Dh, Pi), FNAME_GENERIC, half, 0, int)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(Dh, PU3AS1i), FNAME_GLOBAL, half, 1,
                                  int)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(Dh, PU3AS3i), FNAME_LOCAL, half, 3,
                                  int)
MANUALLY_MANGLED_V_V_VP_VECTORIZE(FUNCNAME(Dh, PU3AS5i), FNAME_PRIVATE, half, 5,
                                  int)

#undef FNAME_GENERIC
#undef FNAME_GLOBAL
#undef FNAME_LOCAL
#undef FNAME_PRIVATE

#endif
