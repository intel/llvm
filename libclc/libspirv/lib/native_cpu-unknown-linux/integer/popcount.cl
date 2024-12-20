#include <clc/clcfunc.h>
#include <clc/clcmacro.h>
#include <libspirv/spirv.h>

// We can't use __builtin_popcountg because it supports only unsigned
// types, and we can't use __builtin_popcount because the implicit cast
// to int doesn't work due to sign extension, so we use type punning to
// preserve the bit pattern and avoid sign extension.

#define DEF_POPCOUNT_HELPER(TYPE, UTYPE) \
_CLC_OVERLOAD TYPE __popcount_helper(TYPE c) { \
  return __builtin_popcountg(*(UTYPE*)&c); \
}

DEF_POPCOUNT_HELPER(char, unsigned char)
DEF_POPCOUNT_HELPER(schar, unsigned char)
DEF_POPCOUNT_HELPER(short, unsigned short)

_CLC_DEFINE_UNARY_BUILTIN(int, __spirv_ocl_popcount, __builtin_popcount, int)
_CLC_DEFINE_UNARY_BUILTIN(uint, __spirv_ocl_popcount, __builtin_popcount, uint)
_CLC_DEFINE_UNARY_BUILTIN(short, __spirv_ocl_popcount, __popcount_helper, short)
_CLC_DEFINE_UNARY_BUILTIN(ushort, __spirv_ocl_popcount, __builtin_popcountg, ushort)
_CLC_DEFINE_UNARY_BUILTIN(long, __spirv_ocl_popcount, __builtin_popcountl, long)
_CLC_DEFINE_UNARY_BUILTIN(ulong, __spirv_ocl_popcount, __builtin_popcountl, ulong)
_CLC_DEFINE_UNARY_BUILTIN(char, __spirv_ocl_popcount, __popcount_helper, char)
_CLC_DEFINE_UNARY_BUILTIN(uchar, __spirv_ocl_popcount, __builtin_popcountg, uchar)
_CLC_DEFINE_UNARY_BUILTIN(schar, __spirv_ocl_popcount, __popcount_helper, schar)
