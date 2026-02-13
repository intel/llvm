//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>
#include <clc/math/clc_sqrt.h>

#define __CLC_FUNCTION __clc_sqrt
#define __CLC_BUILTIN __ocml_sqrt

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
double __ocml_sqrt_f64(double);
#define __CLC_BUILTIN_D __CLC_XCONCAT(__CLC_BUILTIN, _f64)
#endif // cl_khr_fp64

#define __CLC_DOUBLE_ONLY
#include <clc/math/unary_builtin_scalarize.inc>
