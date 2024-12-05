#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <core/clc_core.h>
#include <spirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF char mad_sat(char x, char y, char z) {
  return __clc_mad_sat(x, y, z);
}

_CLC_OVERLOAD _CLC_DEF uchar mad_sat(uchar x, uchar y, uchar z) {
  return __clc_mad_sat(x, y, z);
}

_CLC_OVERLOAD _CLC_DEF short mad_sat(short x, short y, short z) {
  return __clc_mad_sat(x, y, z);
}

_CLC_OVERLOAD _CLC_DEF ushort mad_sat(ushort x, ushort y, ushort z) {
  return __clc_mad_sat(x, y, z);
}

_CLC_OVERLOAD _CLC_DEF int mad_sat(int x, int y, int z) {
  return __clc_mad_sat(x, y, z);
}

_CLC_OVERLOAD _CLC_DEF uint mad_sat(uint x, uint y, uint z) {
  return __clc_mad_sat(x, y, z);
}

_CLC_OVERLOAD _CLC_DEF long mad_sat(long x, long y, long z) {
  return __clc_mad_sat(x, y, z);
}

_CLC_OVERLOAD _CLC_DEF ulong mad_sat(ulong x, ulong y, ulong z) {
  return __clc_mad_sat(x, y, z);
}

_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, char, mad_sat, char, char, char)
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uchar, mad_sat, uchar, uchar, uchar)
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, short, mad_sat, short, short, short)
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, mad_sat, ushort, ushort, ushort)
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, mad_sat, int, int, int)
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, mad_sat, uint, uint, uint)
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, mad_sat, long, long, long)
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ulong, mad_sat, ulong, ulong, ulong)
