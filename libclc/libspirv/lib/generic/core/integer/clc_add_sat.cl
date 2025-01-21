//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <core/clc_core.h>

_CLC_OVERLOAD _CLC_DEF char __clc_add_sat(char x, char y) {
  short r = x + y;
  return __clc_convert_char_sat(r);
}

_CLC_OVERLOAD _CLC_DEF schar __clc_add_sat(schar x, schar y) {
  short r = x + y;
  return __clc_convert_schar_sat(r);
}

_CLC_OVERLOAD _CLC_DEF uchar __clc_add_sat(uchar x, uchar y) {
  ushort r = x + y;
  return __clc_convert_uchar_sat(r);
}

_CLC_OVERLOAD _CLC_DEF short __clc_add_sat(short x, short y) {
  int r = x + y;
  return __clc_convert_short_sat(r);
}

_CLC_OVERLOAD _CLC_DEF ushort __clc_add_sat(ushort x, ushort y) {
  uint r = x + y;
  return __clc_convert_ushort_sat(r);
}

_CLC_OVERLOAD _CLC_DEF int __clc_add_sat(int x, int y) {
  int r;
  if (__builtin_sadd_overflow(x, y, &r))
    // The oveflow can only occur if both are pos or both are neg,
    // thus we only need to check one operand
    return x > 0 ? INT_MAX : INT_MIN;
  return r;
}

_CLC_OVERLOAD _CLC_DEF uint __clc_add_sat(uint x, uint y) {
  uint r;
  if (__builtin_uadd_overflow(x, y, &r))
    return UINT_MAX;
  return r;
}

_CLC_OVERLOAD _CLC_DEF long __clc_add_sat(long x, long y) {
  long r;
  if (__builtin_saddl_overflow(x, y, &r))
    // The oveflow can only occur if both are pos or both are neg,
    // thus we only need to check one operand
    return x > 0 ? LONG_MAX : LONG_MIN;
  return r;
}

_CLC_OVERLOAD _CLC_DEF ulong __clc_add_sat(ulong x, ulong y) {
  ulong r;
  if (__builtin_uaddl_overflow(x, y, &r))
    return ULONG_MAX;
  return r;
}

_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, char, __clc_add_sat, char, char)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, schar, __clc_add_sat, schar,
                      schar)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uchar, __clc_add_sat, uchar,
                      uchar)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, short, __clc_add_sat, short,
                      short)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, __clc_add_sat, ushort,
                      ushort)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, __clc_add_sat, int, int)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, __clc_add_sat, uint, uint)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, __clc_add_sat, long, long)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ulong, __clc_add_sat, ulong,
                      ulong)
