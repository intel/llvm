//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 64
#define __CLC_CONVERT_NATN __CLC_XCONCAT(__spirv_SConvert_Rlong, __CLC_VECSIZE)
#elif __CLC_FPSIZE == 32
#define __CLC_CONVERT_NATN __CLC_XCONCAT(__spirv_SConvert_Rint, __CLC_VECSIZE)
#elif __CLC_FPSIZE == 16
#define __CLC_CONVERT_NATN __CLC_XCONCAT(__spirv_SConvert_Rshort, __CLC_VECSIZE)
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __spirv_ocl_minmag(__CLC_GENTYPE x,
                                                        __CLC_GENTYPE y) {
  const __CLC_GENTYPE res =
      __spirv_ocl_select(y, x,
        __CLC_CONVERT_NATN((__CLC_SCHARN)__spirv_FOrdLessThan(
                             __spirv_ocl_fabs(x), __spirv_ocl_fabs(y))));
  return __spirv_ocl_select(
      res, __spirv_ocl_fmin(x, y),
      __CLC_CONVERT_NATN( (__CLC_SCHARN)(__spirv_IsNan(x) | __spirv_IsNan(y) |
                        __spirv_FOrdEqual(__spirv_ocl_fabs(x),
                                          __spirv_ocl_fabs(y)))));
}

#undef __CLC_CONVERT_NATN
