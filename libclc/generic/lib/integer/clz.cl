#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <spirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF char clz(char x) {
  return __spirv_ocl_clz(x);
}

_CLC_OVERLOAD _CLC_DEF uchar clz(uchar x) {
  return __spirv_ocl_clz(x);
}

_CLC_OVERLOAD _CLC_DEF short clz(short x) {
  return __spirv_ocl_clz(x);
}

_CLC_OVERLOAD _CLC_DEF ushort clz(ushort x) {
  return __spirv_ocl_clz(x);
}

_CLC_OVERLOAD _CLC_DEF int clz(int x) {
  return __spirv_ocl_clz(x);
}

_CLC_OVERLOAD _CLC_DEF uint clz(uint x) {
  return __spirv_ocl_clz(x);
}

_CLC_OVERLOAD _CLC_DEF long clz(long x) {
  return __spirv_ocl_clz(x);
}

_CLC_OVERLOAD _CLC_DEF ulong clz(ulong x) {
  return __spirv_ocl_clz(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, char, clz, char)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uchar, clz, uchar)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, short, clz, short)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, clz, ushort)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, clz, int)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, clz, uint)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, clz, long)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ulong, clz, ulong)
