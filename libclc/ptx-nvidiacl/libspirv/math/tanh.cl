//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

#include "../../include/libdevice.h"
#include <clcmacro.h>

int __clc_nvvm_reflect_arch();
int __clc_nvvm_reflect_approx_tanh();

float __my_tanhf (float x){
  if(__clc_nvvm_reflect_approx_tanh()) {
    return __nvvm_tanh_approx_f(x);
  } else {
    return __nv_tanhf(x);
  }
}

#define __CLC_FUNCTION __spirv_ocl_tanh
#define __CLC_BUILTIN __nv_tanh
#define __CLC_BUILTIN_F __my_tanhf
#include <math/unary_builtin.inc>
