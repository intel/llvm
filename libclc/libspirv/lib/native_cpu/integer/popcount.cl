#include <clc/clcfunc.h>
#include <libspirv/spirv.h>

#define FUNCTION __spirv_ocl_popcount
#define __CLC_SCALAR
#define __CLC_MIN_VECSIZE 1

// We can't use __builtin_popcountg because it supports only unsigned
// types, and we can't use __builtin_popcount because the implicit cast
// to int doesn't work due to sign extension, so we use type punning to
// preserve the bit pattern and avoid sign extension.

#define DEF_POPCOUNT_HELPER(TYPE, UTYPE)                                       \
  _CLC_OVERLOAD TYPE __popcount_helper(TYPE c) {                               \
    return __builtin_popcountg(*(UTYPE *)&c);                                  \
  }

DEF_POPCOUNT_HELPER(char, unsigned char)
DEF_POPCOUNT_HELPER(short, unsigned short)

#define FUNCTION __spirv_ocl_popcount
#define __IMPL_FUNCTION
#define __CLC_BODY <clc/shared/ternary_def_scalarize.inc>
#include <clc/integer/gentype.inc>

#define __CLC_GENTYPE int
#define __IMPL_FUNCTION __builtin_popcount
#include <clc/shared/unary_def_scalarize.inc>
#undef __IMPL_FUNCTION
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint
#define __IMPL_FUNCTION __builtin_popcount
#include <clc/shared/unary_def_scalarize.inc>
#undef __IMPL_FUNCTION
#undef __CLC_GENTYPE

#define __CLC_GENTYPE short
#define __IMPL_FUNCTION __popcount_helper
#include <clc/shared/unary_def_scalarize.inc>
#undef __IMPL_FUNCTION
#undef __CLC_GENTYPE

#define __CLC_GENTYPE ushort
#define __IMPL_FUNCTION __builtin_popcountg
#include <clc/shared/unary_def_scalarize.inc>
#undef __IMPL_FUNCTION
#undef __CLC_GENTYPE

#define __CLC_GENTYPE long
#define __IMPL_FUNCTION __builtin_popcountl
#include <clc/shared/unary_def_scalarize.inc>
#undef __IMPL_FUNCTION
#undef __CLC_GENTYPE

#define __CLC_GENTYPE ulong
#define __IMPL_FUNCTION __builtin_popcountl
#include <clc/shared/unary_def_scalarize.inc>
#undef __IMPL_FUNCTION
#undef __CLC_GENTYPE

#define __CLC_GENTYPE char
#define __IMPL_FUNCTION __popcount_helper
#include <clc/shared/unary_def_scalarize.inc>
#undef __IMPL_FUNCTION
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uchar
#define __IMPL_FUNCTION __builtin_popcountg
#include <clc/shared/unary_def_scalarize.inc>
#undef __IMPL_FUNCTION
#undef __CLC_GENTYPE

#undef __CLC_MIN_VECSIZE
#undef __CLC_SCALAR
#undef FUNCTION
