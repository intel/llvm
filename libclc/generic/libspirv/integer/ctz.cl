//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <spirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF short __spirv_ocl_ctz(short x) {
  return x ? __builtin_ctzs(x) : 16;
}

_CLC_OVERLOAD _CLC_DEF ushort __spirv_ocl_ctz(ushort x) {
  return x ? __builtin_ctzs(x) : 16;
}

_CLC_OVERLOAD _CLC_DEF int __spirv_ocl_ctz(int x) {
  return x ? __builtin_ctz(x) : 32;
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_ocl_ctz(uint x) {
  return x ? __builtin_ctz(x) : 32;
}

_CLC_OVERLOAD _CLC_DEF long __spirv_ocl_ctz(long x) {
  return x ? __builtin_ctzl(x) : 64;
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_ocl_ctz(ulong x) {
  return x ? __builtin_ctzl(x) : 64;
}

_CLC_OVERLOAD _CLC_DEF char __spirv_ocl_ctz(char x) {
  return x ? __spirv_ocl_ctz((ushort)(uchar)x) : 8;
}

_CLC_OVERLOAD _CLC_DEF schar __spirv_ocl_ctz(schar x) {
  return x ? __spirv_ocl_ctz((ushort)(uchar)x) : 8;
}

_CLC_OVERLOAD _CLC_DEF uchar __spirv_ocl_ctz(uchar x) {
  return x ? __spirv_ocl_ctz((ushort)x) : 8;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, char, __spirv_ocl_ctz, char)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, schar, __spirv_ocl_ctz, schar)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uchar, __spirv_ocl_ctz, uchar)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, short, __spirv_ocl_ctz, short)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, __spirv_ocl_ctz, ushort)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, __spirv_ocl_ctz, int)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, __spirv_ocl_ctz, uint)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, __spirv_ocl_ctz, long)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ulong, __spirv_ocl_ctz, ulong)
