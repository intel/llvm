//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

float __ocml_remainder_f32(float,float);
double __ocml_remainder_f64(double,double);

#define __CLC_FUNCTION __spirv_ocl_remainder
#define __CLC_BUILTIN __ocml_remainder
#define __CLC_BUILTIN_F __CLC_XCONCAT(__CLC_BUILTIN, _f32)
#define __CLC_BUILTIN_D __CLC_XCONCAT(__CLC_BUILTIN, _f64)
#include <math/binary_builtin.inc>
