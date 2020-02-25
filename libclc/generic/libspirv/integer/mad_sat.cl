//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include "../../lib/clcmacro.h"

_CLC_OVERLOAD _CLC_DEF char __spirv_ocl_u_mad_sat(char x, char y, char z) {
  return __spirv_ocl_u_clamp((short)__spirv_ocl_u_mad24((short)x, (short)y, (short)z),
      (short)CHAR_MIN, (short) CHAR_MAX);
}

_CLC_OVERLOAD _CLC_DEF uchar __spirv_ocl_u_mad_sat(uchar x, uchar y, uchar z) {
  return __spirv_ocl_u_clamp((ushort)__spirv_ocl_u_mad24((ushort)x, (ushort)y, (ushort)z),
      (ushort)0, (ushort) UCHAR_MAX);
}

_CLC_OVERLOAD _CLC_DEF short __spirv_ocl_u_mad_sat(short x, short y, short z) {
  return __spirv_ocl_u_clamp((int)__spirv_ocl_u_mad24((int)x, (int)y, (int)z),
      (int)SHRT_MIN, (int) SHRT_MAX);
}

_CLC_OVERLOAD _CLC_DEF ushort __spirv_ocl_u_mad_sat(ushort x, ushort y, ushort z) {
  return __spirv_ocl_u_clamp((uint)__spirv_ocl_u_mad24((uint)x, (uint)y, (uint)z),
      (uint)0, (uint) USHRT_MAX);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_ocl_u_mad_sat(int x, int y, int z) {
  int mhi = __spirv_ocl_u_mul_hi(x, y);
  uint mlo = x * y;
  long m = __spirv_ocl_u_upsample(mhi, mlo);
  m += z;
  if (m > INT_MAX)
    return INT_MAX;
  if (m < INT_MIN)
    return INT_MIN;
  return m;
}

_CLC_OVERLOAD _CLC_DEF uint __spirv_ocl_u_mad_sat(uint x, uint y, uint z) {
  if (__spirv_ocl_u_mul_hi(x, y) != 0)
    return UINT_MAX;
  return __spirv_ocl_u_add_sat(x * y, z);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_ocl_u_mad_sat(long x, long y, long z) {
  long hi = __spirv_ocl_u_mul_hi(x, y);
  ulong ulo = x * y;
  long  slo = x * y;
  /* Big overflow of more than 2 bits, add can't fix this */
  if (((x < 0) == (y < 0)) && hi != 0)
    return LONG_MAX;
  /* Low overflow in mul and z not neg enough to correct it */
  if (hi == 0 && ulo >= LONG_MAX && (z > 0 || (ulo + z) > LONG_MAX))
    return LONG_MAX;
  /* Big overflow of more than 2 bits, add can't fix this */
  if (((x < 0) != (y < 0)) && hi != -1)
    return LONG_MIN;
  /* Low overflow in mul and z not pos enough to correct it */
  if (hi == -1 && ulo <= ((ulong)LONG_MAX + 1UL) && (z < 0 || z < (LONG_MAX - ulo)))
    return LONG_MIN;
  /* We have checked all conditions, any overflow in addition returns
   * the correct value */
  return ulo + z;
}

_CLC_OVERLOAD _CLC_DEF ulong __spirv_ocl_u_mad_sat(ulong x, ulong y, ulong z) {
  if (__spirv_ocl_u_mul_hi(x, y) != 0)
    return ULONG_MAX;
  return __spirv_ocl_u_add_sat(x * y, z);
}

_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, char, __spirv_ocl_u_mad_sat, char, char, char)
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uchar, __spirv_ocl_u_mad_sat, uchar, uchar, uchar)
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, short, __spirv_ocl_u_mad_sat, short, short, short)
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, __spirv_ocl_u_mad_sat, ushort, ushort, ushort)
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, __spirv_ocl_u_mad_sat, int, int, int)
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, __spirv_ocl_u_mad_sat, uint, uint, uint)
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, __spirv_ocl_u_mad_sat, long, long, long)
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ulong, __spirv_ocl_u_mad_sat, ulong, ulong, ulong)
