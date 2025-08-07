//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

extern int __clc_nvvm_reflect_ftz();

_CLC_DEF _CLC_OVERLOAD float __spirv_ocl_native_recip(float x) {
  return (__clc_nvvm_reflect_ftz()) ? __nvvm_rcp_approx_ftz_f(x)
                                    : __nvvm_rcp_approx_f(x);
}

#define FUNCTION __spirv_ocl_native_recip
#define __FLOAT_ONLY
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
