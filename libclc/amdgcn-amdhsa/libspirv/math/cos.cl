//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN __clc_fp32_t
__spirv_ocl_cos(__clc_fp32_t In) {
  return __builtin_amdgcn_cosf(In);
}
