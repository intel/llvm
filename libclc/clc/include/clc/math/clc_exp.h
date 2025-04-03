//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_MATH_CLC_EXP_H__
#define __CLC_MATH_CLC_EXP_H__

<<<<<<<< HEAD:libclc/libspirv/lib/generic/math/exp_helper.h
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
_CLC_DECL double __clc_exp_helper(double x, double x_min, double x_max,
                                  double r, int n);
========
#define __CLC_BODY <clc/math/unary_decl.inc>
#define __CLC_FUNCTION __clc_exp
>>>>>>>> f14ff59da7f98a405999bcc8481b20446de0d0cd:libclc/clc/include/clc/math/clc_exp.h

#include <clc/math/gentype.inc>

#undef __CLC_BODY
#undef __CLC_FUNCTION

#endif // __CLC_MATH_CLC_EXP_H__
