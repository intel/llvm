//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_VECSIZE_OR_1 >= 8

_CLC_OVERLOAD _CLC_DEF __CLC_SCALAR_GENTYPE __spirv_Dot(__CLC_GENTYPE p0,
                                                        __CLC_GENTYPE p1) {
  return __spirv_Dot(p0.lo, p1.lo) + __spirv_Dot(p0.hi, p1.hi);
}

#elif __CLC_VECSIZE_OR_1 >= 2

_CLC_OVERLOAD _CLC_DEF __CLC_SCALAR_GENTYPE __spirv_Dot(__CLC_GENTYPE p0,
                                                        __CLC_GENTYPE p1) {
  return __clc_dot(p0, p1);
}

#endif
