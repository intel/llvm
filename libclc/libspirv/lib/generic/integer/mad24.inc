//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_GENSIZE == 32

#ifdef __CLC_GEN_S
#define DECL __spirv_ocl_s_mad24
#else
#define DECL __spirv_ocl_u_mad24
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE DECL(__CLC_GENTYPE x, __CLC_GENTYPE y,
                                          __CLC_GENTYPE z) {
  return __clc_mad24(x, y, z);
}

#undef DECL

#endif // __CLC_GENSIZE == 32
