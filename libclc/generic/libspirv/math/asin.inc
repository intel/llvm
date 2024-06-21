//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 64
#define __CLC_CONST(x) x
#elif __CLC_FPSIZE == 32
#define __CLC_CONST(x) x##f
#elif __CLC_FPSIZE == 16
#define __CLC_CONST(x) x##h
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __spirv_ocl_asin(__CLC_GENTYPE x) {
  return __spirv_ocl_atan2(
      x, __spirv_ocl_sqrt((__CLC_GENTYPE)__CLC_CONST(1.0) - (x * x)));
}

#undef __CLC_CONST
