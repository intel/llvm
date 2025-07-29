//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_GEN_S
#define DECL __spirv_ocl_s_abs_diff
#else
#define DECL __spirv_ocl_u_abs_diff
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_U_GENTYPE DECL(__CLC_GENTYPE x, __CLC_GENTYPE y) {
  return __clc_abs_diff(x, y);
}

#undef DECL
