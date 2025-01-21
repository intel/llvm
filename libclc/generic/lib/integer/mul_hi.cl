#include <clc/clc.h>
#include <core/clc_core.h>

#define __CLC_MUL_HI_IMPL(BGENTYPE, GENTYPE, GENSIZE)                          \
  _CLC_OVERLOAD _CLC_DEF GENTYPE mul_hi(GENTYPE x, GENTYPE y) {                \
    return __clc_mul_hi(x, y);                                                 \
  }

_CLC_OVERLOAD _CLC_DEF long mul_hi(long x, long y){
  return __clc_mul_hi(x, y);
}

_CLC_OVERLOAD _CLC_DEF ulong mul_hi(ulong x, ulong y){
  return __clc_mul_hi(x, y);
}

#define __CLC_MUL_HI_VEC(GENTYPE)                                              \
  _CLC_OVERLOAD _CLC_DEF GENTYPE##2 mul_hi(GENTYPE##2 x, GENTYPE##2 y) {       \
    return __clc_mul_hi(x, y);                                                 \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF GENTYPE##3 mul_hi(GENTYPE##3 x, GENTYPE##3 y) {       \
    return __clc_mul_hi(x, y);                                                 \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF GENTYPE##4 mul_hi(GENTYPE##4 x, GENTYPE##4 y) {       \
    return __clc_mul_hi(x, y);                                                 \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF GENTYPE##8 mul_hi(GENTYPE##8 x, GENTYPE##8 y) {       \
    return __clc_mul_hi(x, y);                                                 \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF GENTYPE##16 mul_hi(GENTYPE##16 x, GENTYPE##16 y) {    \
    return __clc_mul_hi(x, y);                                                 \
  }

#define __CLC_MUL_HI_DEC_IMPL(BTYPE, TYPE, BITS) \
    __CLC_MUL_HI_IMPL(BTYPE, TYPE, BITS) \
    __CLC_MUL_HI_VEC(TYPE)

#define __CLC_MUL_HI_TYPES() \
    __CLC_MUL_HI_DEC_IMPL(short, char, 8) \
    __CLC_MUL_HI_DEC_IMPL(ushort, uchar, 8) \
    __CLC_MUL_HI_DEC_IMPL(int, short, 16) \
    __CLC_MUL_HI_DEC_IMPL(uint, ushort, 16) \
    __CLC_MUL_HI_DEC_IMPL(long, int, 32) \
    __CLC_MUL_HI_DEC_IMPL(ulong, uint, 32) \
    __CLC_MUL_HI_VEC(long) \
    __CLC_MUL_HI_VEC(ulong)

__CLC_MUL_HI_TYPES()

#undef __CLC_MUL_HI_TYPES
#undef __CLC_MUL_HI_DEC_IMPL
#undef __CLC_MUL_HI_IMPL
#undef __CLC_MUL_HI_VEC
#undef __CLC_B32
