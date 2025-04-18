//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/*
 * There are multiple formulas for calculating arccosine of x:
 * 1) acos(x) = (1/2*pi) + i * ln(i*x + sqrt(1-x^2)) (notice the 'i'...)
 * 2) acos(x) = pi/2 + asin(-x) (asin isn't implemented yet)
 * 3) acos(x) = pi/2 - asin(x) (ditto)
 * 4) acos(x) = 2*atan2(sqrt(1-x), sqrt(1+x))
 * 5) acos(x) = pi/2 - atan2(x, ( sqrt(1-x^2) ) )
 *
 * Options 1-3 are not currently usable, #5 generates more concise radeonsi
 * bitcode and assembly than #4 (134 vs 132 instructions on radeonsi), but
 * precision of #4 may be better.
 */

#if __CLC_FPSIZE == 64
#define __CLC_CONST(x) x
#elif __CLC_FPSIZE == 32
#define __CLC_CONST(x) x##f
#elif __CLC_FPSIZE == 16
#define __CLC_CONST(x) x##h
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __spirv_ocl_acos(__CLC_GENTYPE x) {
  return (
      (__CLC_GENTYPE)__CLC_CONST(2.0) *
      __spirv_ocl_atan2(__spirv_ocl_sqrt((__CLC_GENTYPE)__CLC_CONST(1.0) - x),
                        __spirv_ocl_sqrt((__CLC_GENTYPE)__CLC_CONST(1.0) + x)));
}

#undef __CLC_CONST
