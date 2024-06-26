#include <func.h>
#include <clcmacro.h>
#include <spirv/spirv.h>

_CLC_DEFINE_UNARY_BUILTIN(int, __spirv_ocl_popcount, __builtin_popcount, int)
_CLC_DEFINE_UNARY_BUILTIN(uint, __spirv_ocl_popcount, __builtin_popcount, uint)
_CLC_DEFINE_UNARY_BUILTIN(short, __spirv_ocl_popcount, __builtin_popcount, short)
_CLC_DEFINE_UNARY_BUILTIN(ushort, __spirv_ocl_popcount, __builtin_popcount, ushort)
_CLC_DEFINE_UNARY_BUILTIN(long, __spirv_ocl_popcount, __builtin_popcount, long)
_CLC_DEFINE_UNARY_BUILTIN(ulong, __spirv_ocl_popcount, __builtin_popcount, ulong)
_CLC_DEFINE_UNARY_BUILTIN(char, __spirv_ocl_popcount, __builtin_popcount, char)
_CLC_DEFINE_UNARY_BUILTIN(uchar, __spirv_ocl_popcount, __builtin_popcount, uchar)
_CLC_DEFINE_UNARY_BUILTIN(schar, __spirv_ocl_popcount, __builtin_popcount, schar)
