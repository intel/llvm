//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clcmacro.h>
#include <spirv/spirv.h>
 
double __ocml_fdim_f64(double,double);
float __ocml_fdim_f32(float,float);
 
#define __CLC_FUNCTION __spirv_ocl_fdim
#define __CLC_BUILTIN __ocml_fdim
#define __CLC_BUILTIN_F __CLC_XCONCAT(__CLC_BUILTIN, _f32)
#define __CLC_BUILTIN_D __CLC_XCONCAT(__CLC_BUILTIN, _f64)
#include <math/binary_builtin.inc>
