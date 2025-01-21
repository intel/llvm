//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <libspirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF char __spirv_ocl_clz(char x) {
  return __spirv_ocl_clz((ushort)(uchar)x) - 8;
}

_CLC_OVERLOAD _CLC_DEF schar __spirv_ocl_clz(schar x) {
  return __spirv_ocl_clz((ushort)(uchar)x) - 8;
}

_CLC_OVERLOAD _CLC_DEF uchar __spirv_ocl_clz(uchar x) {
  return __spirv_ocl_clz((ushort)x) - 8;
}

_CLC_OVERLOAD _CLC_DEF short __spirv_ocl_clz(short x) {
  return x ? __builtin_clzs(x) : 16;
}

_CLC_OVERLOAD _CLC_DEF ushort __spirv_ocl_clz(ushort x) {
  return x ? __builtin_clzs(x) : 16;
}

_CLC_OVERLOAD _CLC_DEF int __spirv_ocl_clz(int x) {
  return x ? __builtin_clz(x) : 32;
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_ocl_clz(uint x) {
  return x ? __builtin_clz(x) : 32;
}

_CLC_OVERLOAD _CLC_DEF long __spirv_ocl_clz(long x) {
  return x ? __builtin_clzl(x) : 64;
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_ocl_clz(ulong x) {
  return x ? __builtin_clzl(x) : 64;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, char, __spirv_ocl_clz, char)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, schar, __spirv_ocl_clz, schar)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uchar, __spirv_ocl_clz, uchar)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, short, __spirv_ocl_clz, short)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, __spirv_ocl_clz, ushort)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, __spirv_ocl_clz, int)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, __spirv_ocl_clz, uint)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, __spirv_ocl_clz, long)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ulong, __spirv_ocl_clz, ulong)
