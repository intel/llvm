//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include "../../lib/clcmacro.h"

_CLC_OVERLOAD _CLC_DEF char __spirv_ocl_u_sub_sat(char x, char y) {
  short r = x - y;
  return __spirv_SConvert_Rchar(r);
}

_CLC_OVERLOAD _CLC_DEF uchar __spirv_ocl_u_sub_sat(uchar x, uchar y) {
  short r = x - y;
  return __spirv_SatConvertSToU_Rushort(r);
}

_CLC_OVERLOAD _CLC_DEF short __spirv_ocl_u_sub_sat(short x, short y) {
  int r = x - y;
  return __spirv_SConvert_Rshort(r);
}

_CLC_OVERLOAD _CLC_DEF ushort __spirv_ocl_u_sub_sat(ushort x, ushort y) {
  int r = x - y;
  return __spirv_SatConvertSToU_Rushort(r);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_ocl_u_sub_sat(int x, int y) {
  int r;
  if (__builtin_ssub_overflow(x, y, &r))
    // The oveflow can only occur in the direction of the first operand
    return x > 0 ? INT_MAX : INT_MIN;
  return r;
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_ocl_u_sub_sat(uint x, uint y) {
  uint r;
  if (__builtin_usub_overflow(x, y, &r))
	return 0;
  return r;
}

_CLC_OVERLOAD _CLC_DEF long __spirv_ocl_u_sub_sat(long x, long y) {
  long r;
  if (__builtin_ssubl_overflow(x, y, &r))
    // The oveflow can only occur in the direction of the first operand
    return x > 0 ? LONG_MAX : LONG_MIN;
  return r;
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_ocl_u_sub_sat(ulong x, ulong y) {
  ulong r;
  if (__builtin_usubl_overflow(x, y, &r))
	return 0;
  return r;
}

_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, char, __spirv_ocl_u_sub_sat, char, char)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uchar, __spirv_ocl_u_sub_sat, uchar, uchar)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, short, __spirv_ocl_u_sub_sat, short, short)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, __spirv_ocl_u_sub_sat, ushort, ushort)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, __spirv_ocl_u_sub_sat, int, int)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, __spirv_ocl_u_sub_sat, uint, uint)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, __spirv_ocl_u_sub_sat, long, long)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ulong, __spirv_ocl_u_sub_sat, ulong, ulong)
