//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __spirv_ocl_maxmag(__CLC_GENTYPE x,
                                                        __CLC_GENTYPE y) {
  const __CLC_GENTYPE res =
      __spirv_ocl_select(y, x,
                         __CLC_CONVERT_BIT_INTN(__spirv_FOrdGreaterThan(
                             __spirv_ocl_fabs(x), __spirv_ocl_fabs(y))));
  return __spirv_ocl_select(
      res, __spirv_ocl_fmax(x, y),
      __CLC_CONVERT_BIT_INTN(
          (__spirv_IsNan(x) | __spirv_IsNan(y) |
           __spirv_FOrdEqual(__spirv_ocl_fabs(x), __spirv_ocl_fabs(y)))));
}
