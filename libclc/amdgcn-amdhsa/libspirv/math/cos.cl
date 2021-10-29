//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//#include <spirv/spirv.h>

//_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
//__spirv_ocl_cos(__clc_fp32_t In) {
//  return __builtin_amdgcn_cosf(In);
//}

 #include <spirv/spirv.h>
 #include <clcmacro.h>
 double __ocml_cos_f64(double);
 float __ocml_cos_f32(float);
 #define __CLC_FUNCTION __spirv_ocl_cos
 #define __CLC_BUILTIN __ocml_cos
 #define __CLC_BUILTIN_F __CLC_XCONCAT(__CLC_BUILTIN, _f32)
 #define __CLC_BUILTIN_D __CLC_XCONCAT(__CLC_BUILTIN, _f64)
 #include <math/unary_builtin.inc>
