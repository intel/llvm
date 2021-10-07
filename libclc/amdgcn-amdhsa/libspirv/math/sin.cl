//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

float __ocml_sin_f32(float);
double __ocml_sin_f64(double);

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
__spirv_ocl_sin(__clc_fp32_t In) {
  return __ocml_sin_f32(In);
}

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp64_t
__spirv_ocl_sin(__clc_fp64_t In) {
  return __ocml_sin_f64(In);
}
