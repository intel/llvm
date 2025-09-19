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

float __ocml_sqrt_f32(float);
#define __CLC_BUILTIN_F __CLC_XCONCAT(__CLC_BUILTIN, _f32)

#define __CLC_FLOAT_ONLY
#include <clc/math/unary_builtin_scalarize.inc>

#undef __CLC_FLOAT_ONLY
#undef __CLC_BUILTIN_H

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
half __ocml_sqrt_f16(half);
#define __CLC_BUILTIN_H __CLC_XCONCAT(__CLC_BUILTIN, _f16)
#endif // cl_khr_fp16

#define __CLC_HALF_ONLY
#include <clc/math/unary_builtin_scalarize.inc>
