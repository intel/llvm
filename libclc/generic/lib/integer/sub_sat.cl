#include <spirv/spirv.h>
#include <clc/clc.h>
#include "../clcmacro.h"

_CLC_OVERLOAD _CLC_DEF char sub_sat(char x, char y) {
  return __spirv_ocl_u_sub_sat(x, y);
}

_CLC_OVERLOAD _CLC_DEF uchar sub_sat(uchar x, uchar y) {
  return __spirv_ocl_u_sub_sat(x, y);
}

_CLC_OVERLOAD _CLC_DEF short sub_sat(short x, short y) {
  return __spirv_ocl_u_sub_sat(x, y);
}

_CLC_OVERLOAD _CLC_DEF ushort sub_sat(ushort x, ushort y) {
  return __spirv_ocl_u_sub_sat(x, y);
}

_CLC_OVERLOAD _CLC_DEF int sub_sat(int x, int y) {
  return __spirv_ocl_u_sub_sat(x, y);
}

_CLC_OVERLOAD _CLC_DEF uint sub_sat(uint x, uint y) {
  return __spirv_ocl_u_sub_sat(x, y);
}

_CLC_OVERLOAD _CLC_DEF long sub_sat(long x, long y) {
  return __spirv_ocl_u_sub_sat(x, y);
}

_CLC_OVERLOAD _CLC_DEF ulong sub_sat(ulong x, ulong y) {
  return __spirv_ocl_u_sub_sat(x, y);
}

_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, char, sub_sat, char, char)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uchar, sub_sat, uchar, uchar)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, short, sub_sat, short, short)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, sub_sat, ushort, ushort)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, sub_sat, int, int)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, sub_sat, uint, uint)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, sub_sat, long, long)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ulong, sub_sat, ulong, ulong)
