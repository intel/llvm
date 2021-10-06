//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

#define SELF __builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0))
#define SUBGROUP_SIZE __spirv_SubgroupMaxSize()

// Shuffle
// int __spirv_SubgroupShuffleINTEL<int>(int, unsigned int)
_CLC_DEF int
_Z28__spirv_SubgroupShuffleINTELIiET_S0_j(int Data, unsigned int InvocationId) {
  int self = SELF;
  int index = InvocationId + (self & ~(SUBGROUP_SIZE - 1));
  return __builtin_amdgcn_ds_bpermute(index << 2, Data);
}

// Sub 32-bit types.
// _Z28__spirv_SubgroupShuffleINTELIaET_S0_j - char
// _Z28__spirv_SubgroupShuffleINTELIhET_S0_j - unsigned char
// _Z28__spirv_SubgroupShuffleINTELIsET_S0_j - long
// _Z28__spirv_SubgroupShuffleINTELItET_S0_j - unsigned long
#define __AMDGCN_CLC_SUBGROUP_SUB_I32(TYPE, MANGLED_TYPE_NAME)                 \
  _CLC_DEF TYPE _Z28__spirv_SubgroupShuffleINTELI##MANGLED_TYPE_NAME##ET_S0_j( \
      TYPE Data, unsigned int InvocationId) {                                  \
    return _Z28__spirv_SubgroupShuffleINTELIiET_S0_j(Data, InvocationId);      \
  }
__AMDGCN_CLC_SUBGROUP_SUB_I32(char, a);
__AMDGCN_CLC_SUBGROUP_SUB_I32(unsigned char, h);
__AMDGCN_CLC_SUBGROUP_SUB_I32(short, s);
__AMDGCN_CLC_SUBGROUP_SUB_I32(unsigned short, t);
#undef __AMDGCN_CLC_SUBGROUP_SUB_I32

// 32-bit types.
// __spirv_SubgroupShuffleINTEL - unsigned int
// __spirv_SubgroupShuffleINTEL-  float
#define __AMDGCN_CLC_SUBGROUP_I32(TYPE, CAST_TYPE, MANGLED_TYPE_NAME)          \
  _CLC_DEF TYPE _Z28__spirv_SubgroupShuffleINTELI##MANGLED_TYPE_NAME##ET_S0_j( \
      TYPE Data, unsigned int InvocationId) {                                  \
    return __builtin_astype(                                                   \
        _Z28__spirv_SubgroupShuffleINTELIiET_S0_j(as_int(Data), InvocationId), \
        CAST_TYPE);                                                            \
  }
__AMDGCN_CLC_SUBGROUP_I32(unsigned int, uint, j);
__AMDGCN_CLC_SUBGROUP_I32(float, float, f);
#undef __AMDGCN_CLC_SUBGROUP_I32

// 64-bit types.
// __spirv_SubgroupShuffleINTEL - long
// __spirv_SubgroupShuffleINTEL - unsigned long
// __spirv_SubgroupShuffleINTEL - double
#define __AMDGCN_CLC_SUBGROUP_I64(TYPE, CAST_TYPE, MANGLED_TYPE_NAME)          \
  _CLC_DEF TYPE _Z28__spirv_SubgroupShuffleINTELI##MANGLED_TYPE_NAME##ET_S0_j( \
      TYPE Data, unsigned int InvocationId) {                                  \
    int2 tmp = as_int2(Data);                                                  \
    tmp.lo = _Z28__spirv_SubgroupShuffleINTELIiET_S0_j(tmp.lo, InvocationId);  \
    tmp.hi = _Z28__spirv_SubgroupShuffleINTELIiET_S0_j(tmp.hi, InvocationId);  \
    return __builtin_astype(tmp, CAST_TYPE);                                   \
  }
__AMDGCN_CLC_SUBGROUP_I64(long, long, l);
__AMDGCN_CLC_SUBGROUP_I64(unsigned long, ulong, m);
__AMDGCN_CLC_SUBGROUP_I64(double, double, d);
#undef __AMDGCN_CLC_SUBGROUP_I64

// Vector types.
#define __AMDGCN_CLC_SUBGROUP_TO_VEC(TYPE, MANGLED_SCALAR_TY, NUM_ELEMS)             \
  _CLC_DEF TYPE                                                                      \
      _Z28__spirv_SubgroupShuffleINTELIDv##NUM_ELEMS##_##MANGLED_SCALAR_TY##ET_S1_j( \
          TYPE Data, unsigned int InvocationId) {                                    \
    TYPE res;                                                                        \
    for (int i = 0; i < NUM_ELEMS; ++i) {                                            \
      res[i] = _Z28__spirv_SubgroupShuffleINTELI##MANGLED_SCALAR_TY##ET_S0_j(        \
          Data[i], InvocationId);                                                    \
    }                                                                                \
    return res;                                                                      \
  }

// [u]char
__AMDGCN_CLC_SUBGROUP_TO_VEC(char2, a, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(char4, a, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(char8, a, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(char16, a, 16)
__AMDGCN_CLC_SUBGROUP_TO_VEC(uchar2, h, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(uchar4, h, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(uchar8, h, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(uchar16, h, 16)
// [u]short
__AMDGCN_CLC_SUBGROUP_TO_VEC(short2, s, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(short4, s, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(short8, s, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(short16, s, 16)
__AMDGCN_CLC_SUBGROUP_TO_VEC(ushort2, t, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(ushort4, t, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(ushort8, t, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(ushort16, t, 16)
// [u]int
__AMDGCN_CLC_SUBGROUP_TO_VEC(int2, i, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(int4, i, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(int8, i, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(int16, i, 16)
__AMDGCN_CLC_SUBGROUP_TO_VEC(uint2, j, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(uint4, j, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(uint8, j, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(uint16, j, 16)
// [u]long
__AMDGCN_CLC_SUBGROUP_TO_VEC(long2, l, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(long4, l, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(long8, l, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(long16, l, 16)
__AMDGCN_CLC_SUBGROUP_TO_VEC(ulong2, m, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(ulong4, m, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(ulong8, m, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(ulong16, m, 16)
// float
__AMDGCN_CLC_SUBGROUP_TO_VEC(float2, f, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(float4, f, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(float8, f, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(float16, f, 16)
// double
__AMDGCN_CLC_SUBGROUP_TO_VEC(double2, d, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(double4, d, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(double8, d, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(double16, d, 16)
#undef __AMDGCN_CLC_SUBGROUP_TO_VEC

// Shuffle XOR
// int __spirv_SubgroupShuffleXorINTEL<int>(int, unsigned int)
_CLC_DEF int
_Z31__spirv_SubgroupShuffleXorINTELIiET_S0_j(int Data,
                                             unsigned int InvocationId) {
  int self = SELF;
  unsigned int index = self ^ InvocationId;
  index =
      index >= ((self + SUBGROUP_SIZE) & ~(SUBGROUP_SIZE - 1)) ? self : index;
  return __builtin_amdgcn_ds_bpermute(index << 2, Data);
}

// Sub 32-bit types.
// _Z31__spirv_SubgroupShuffleXorINTELIaET_S0_j - char
// _Z31__spirv_SubgroupShuffleXorINTELIhET_S0_j - unsigned char
// _Z31__spirv_SubgroupShuffleXorINTELIsET_S0_j - short
// _Z31__spirv_SubgroupShuffleXorINTELItET_S0_j - unsigned short
#define __AMDGCN_CLC_SUBGROUP_XOR_SUB_I32(TYPE, MANGLED_TYPE_NAME)             \
  _CLC_DEF TYPE                                                                \
      _Z31__spirv_SubgroupShuffleXorINTELI##MANGLED_TYPE_NAME##ET_S0_j(        \
          TYPE Data, unsigned int InvocationId) {                              \
    return _Z31__spirv_SubgroupShuffleXorINTELIiET_S0_j(Data, InvocationId);   \
  }
__AMDGCN_CLC_SUBGROUP_XOR_SUB_I32(char, a);
__AMDGCN_CLC_SUBGROUP_XOR_SUB_I32(unsigned char, h);
__AMDGCN_CLC_SUBGROUP_XOR_SUB_I32(short, s);
__AMDGCN_CLC_SUBGROUP_XOR_SUB_I32(unsigned short, t);
#undef __AMDGCN_CLC_SUBGROUP_XOR_SUB_I32

// 32-bit types.
// __spirv_SubgroupShuffleXorINTEL - unsigned int
// __spirv_SubgroupShuffleXorINTEL - float
#define __AMDGCN_CLC_SUBGROUP_XOR_I32(TYPE, CAST_TYPE, MANGLED_TYPE_NAME)      \
  _CLC_DEF TYPE                                                                \
      _Z31__spirv_SubgroupShuffleXorINTELI##MANGLED_TYPE_NAME##ET_S0_j(        \
          TYPE Data, unsigned int InvocationId) {                              \
    return __builtin_astype(_Z31__spirv_SubgroupShuffleXorINTELIiET_S0_j(      \
                                as_int(Data), InvocationId),                   \
                            CAST_TYPE);                                        \
  }
__AMDGCN_CLC_SUBGROUP_XOR_I32(unsigned int, uint, j);
__AMDGCN_CLC_SUBGROUP_XOR_I32(float, float, f);
#undef __AMDGCN_CLC_SUBGROUP_XOR_I32

// 64-bit types.
// __spirv_SubgroupShuffleXorINTEL - long
// __spirv_SubgroupShuffleXorINTEL - unsigned long
// __spirv_SubgroupShuffleXorINTEL - double
#define __AMDGCN_CLC_SUBGROUP_XOR_I64(TYPE, CAST_TYPE, MANGLED_TYPE_NAME)      \
  _CLC_DEF TYPE                                                                \
      _Z31__spirv_SubgroupShuffleXorINTELI##MANGLED_TYPE_NAME##ET_S0_j(        \
          TYPE Data, unsigned int InvocationId) {                              \
    int2 tmp = as_int2(Data);                                                  \
    tmp.lo =                                                                   \
        _Z31__spirv_SubgroupShuffleXorINTELIiET_S0_j(tmp.lo, InvocationId);    \
    tmp.hi =                                                                   \
        _Z31__spirv_SubgroupShuffleXorINTELIiET_S0_j(tmp.hi, InvocationId);    \
    return __builtin_astype(tmp, CAST_TYPE);                                   \
  }
__AMDGCN_CLC_SUBGROUP_XOR_I64(long, long, l);
__AMDGCN_CLC_SUBGROUP_XOR_I64(unsigned long, ulong, m);
__AMDGCN_CLC_SUBGROUP_XOR_I64(double, double, d);
#undef __AMDGCN_CLC_SUBGROUP_XOR_I64

// Vector types.
#define __AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(TYPE, MANGLED_SCALAR_TY, NUM_ELEMS)            \
  _CLC_DEF TYPE                                                                         \
      _Z31__spirv_SubgroupShuffleXorINTELIDv##NUM_ELEMS##_##MANGLED_SCALAR_TY##ET_S1_j( \
          TYPE Data, unsigned int InvocationId) {                                       \
    TYPE res;                                                                           \
    for (int i = 0; i < NUM_ELEMS; ++i) {                                               \
      res[i] =                                                                          \
          _Z31__spirv_SubgroupShuffleXorINTELI##MANGLED_SCALAR_TY##ET_S0_j(             \
              Data[i], InvocationId);                                                   \
    }                                                                                   \
    return res;                                                                         \
  }
// [u]char
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(char2, a, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(char4, a, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(char8, a, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(char16, a, 16)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(uchar2, h, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(uchar4, h, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(uchar8, h, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(uchar16, h, 16)
// [u]short
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(short2, s, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(short4, s, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(short8, s, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(short16, s, 16)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(ushort2, t, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(ushort4, t, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(ushort8, t, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(ushort16, t, 16)
// [u]int
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(int2, i, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(int4, i, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(int8, i, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(int16, i, 16)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(uint2, j, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(uint4, j, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(uint8, j, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(uint16, j, 16)
// [u]long
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(long2, l, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(long4, l, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(long8, l, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(long16, l, 16)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(ulong2, m, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(ulong4, m, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(ulong8, m, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(ulong16, m, 16)
// float
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(float2, f, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(float4, f, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(float8, f, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(float16, f, 16)
// double
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(double2, d, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(double4, d, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(double8, d, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(double16, d, 16)
#undef __AMDGCN_CLC_SUBGROUP_XOR_TO_VEC

// Shuffle Up
// int __spirv_SubgroupShuffleUpINTEL<int>(int, int, unsigned int)
_CLC_DEF int
_Z30__spirv_SubgroupShuffleUpINTELIiET_S0_S0_j(int var, int lane_delta,
                                               unsigned int width) {
  int self = SELF;
  int index = self - lane_delta;
  index = (index < (self & ~(width - 1))) ? index : self;
  return __builtin_amdgcn_ds_bpermute(index << 2, var);
}

// Sub 32-bit types.
// _Z30__spirv_SubgroupShuffleUpINTELIaET_S0_S0_j - char
// _Z30__spirv_SubgroupShuffleUpINTELIhET_S0_S0_j - unsigned char
// _Z30__spirv_SubgroupShuffleUpINTELIsET_S0_S0_j - short
// _Z30__spirv_SubgroupShuffleUpINTELItET_S0_S0_j - unsigned short
#define __AMDGCN_CLC_SUBGROUP_UP_SUB_I32(TYPE, MANGLED_TYPE_NAME)              \
  _CLC_DEF TYPE                                                                \
      _Z30__spirv_SubgroupShuffleUpINTELI##MANGLED_TYPE_NAME##ET_S0_S0_j(      \
          TYPE var, TYPE lane_delta, unsigned int width) {                     \
    return _Z30__spirv_SubgroupShuffleUpINTELIiET_S0_S0_j(var, lane_delta,     \
                                                          width);              \
  }
__AMDGCN_CLC_SUBGROUP_UP_SUB_I32(char, a);
__AMDGCN_CLC_SUBGROUP_UP_SUB_I32(unsigned char, h);
__AMDGCN_CLC_SUBGROUP_UP_SUB_I32(short, s);
__AMDGCN_CLC_SUBGROUP_UP_SUB_I32(unsigned short, t);
#undef __AMDGCN_CLC_SUBGROUP_UP_SUB_I32

// 32-bit types.
// __spirv_SubgroupShuffleUpINTELi - unsigned int
// __spirv_SubgroupShuffleUpINTELi - float
#define __AMDGCN_CLC_SUBGROUP_UP_I32(TYPE, CAST_TYPE, MANGLED_TYPE_NAME)       \
  _CLC_DEF TYPE                                                                \
      _Z30__spirv_SubgroupShuffleUpINTELI##MANGLED_TYPE_NAME##ET_S0_S0_j(      \
          TYPE var, TYPE lane_delta, unsigned int width) {                     \
    return __builtin_astype(_Z30__spirv_SubgroupShuffleUpINTELIiET_S0_S0_j(    \
                                as_int(var), as_int(lane_delta), width),       \
                            CAST_TYPE);                                        \
  }
__AMDGCN_CLC_SUBGROUP_UP_I32(unsigned int, uint, j);
__AMDGCN_CLC_SUBGROUP_UP_I32(float, float, f);
#undef __AMDGCN_CLC_SUBGROUP_UP_I32

// 64-bit types.
// __spirv_SubgroupShuffleUpINTEL - long
// __spirv_SubgroupShuffleUpINTEL - unsigned long
// __spirv_SubgroupShuffleUpINTEL - double
#define __AMDGCN_CLC_SUBGROUP_UP_I64(TYPE, CAST_TYPE, MANGLED_TYPE_NAME)       \
  _CLC_DEF TYPE                                                                \
      _Z30__spirv_SubgroupShuffleUpINTELI##MANGLED_TYPE_NAME##ET_S0_S0_j(      \
          TYPE var, TYPE lane_delta, unsigned int width) {                     \
    int2 tmp = as_int2(var);                                                   \
    tmp.lo = _Z30__spirv_SubgroupShuffleUpINTELIiET_S0_S0_j(                   \
        tmp.lo, (int)lane_delta, width);                                       \
    tmp.hi = _Z30__spirv_SubgroupShuffleUpINTELIiET_S0_S0_j(                   \
        tmp.hi, (int)lane_delta, width);                                       \
    return __builtin_astype(tmp, CAST_TYPE);                                   \
  }
__AMDGCN_CLC_SUBGROUP_UP_I64(long, long, l);
__AMDGCN_CLC_SUBGROUP_UP_I64(unsigned long, ulong, m);
__AMDGCN_CLC_SUBGROUP_UP_I64(double, double, d);
#undef __AMDGCN_CLC_SUBGROUP_UP_I64

// Vector types.
#define __AMDGCN_CLC_SUBGROUP_UP_TO_VEC(TYPE, MANGLED_SCALAR_TY, NUM_ELEMS)               \
  _CLC_DEF TYPE                                                                           \
      _Z30__spirv_SubgroupShuffleUpINTELIDv##NUM_ELEMS##_##MANGLED_SCALAR_TY##ET_S1_S1_j( \
          TYPE var, TYPE lane_delta, unsigned int width) {                                \
    TYPE res;                                                                             \
    for (int i = 0; i < NUM_ELEMS; ++i) {                                                 \
      res[i] =                                                                            \
          _Z30__spirv_SubgroupShuffleUpINTELI##MANGLED_SCALAR_TY##ET_S0_S0_j(             \
              var[i], (unsigned int)lane_delta[0], width);                                \
    }                                                                                     \
    return res;                                                                           \
  }
// [u]char
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(char2, a, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(char4, a, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(char8, a, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(char16, a, 16)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(uchar2, h, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(uchar4, h, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(uchar8, h, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(uchar16, h, 16)
// [u]short
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(short2, s, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(short4, s, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(short8, s, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(short16, s, 16)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(ushort2, t, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(ushort4, t, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(ushort8, t, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(ushort16, t, 16)
// [u]int
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(int2, i, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(int4, i, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(int8, i, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(int16, i, 16)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(uint2, j, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(uint4, j, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(uint8, j, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(uint16, j, 16)
// [u]long
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(long2, l, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(long4, l, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(long8, l, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(long16, l, 16)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(ulong2, m, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(ulong4, m, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(ulong8, m, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(ulong16, m, 16)
// float
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(float2, f, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(float4, f, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(float8, f, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(float16, f, 16)
// double
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(double2, d, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(double4, d, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(double8, d, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(double16, d, 16)
#undef __AMDGCN_CLC_SUBGROUP_UP_TO_VEC

// Shuffle Down
// int __spirv_SubgroupShuffleDownINTEL<int>(int, int, unsigned int)
_CLC_DEF int
_Z32__spirv_SubgroupShuffleDownINTELIiET_S0_S0_j(int var, int lane_delta,
                                                 unsigned int width) {
  unsigned int self = SELF;
  unsigned int index = self + lane_delta;
  index = as_uint(((self & (width - 1)) + lane_delta)) >= width ? self : index;
  return __builtin_amdgcn_ds_bpermute(index << 2, var);
}

// Sub 32-bit types.
// _Z32__spirv_SubgroupShuffleDownINTELIaET_S0_S0_j - char
// _Z32__spirv_SubgroupShuffleDownINTELIhET_S0_S0_j - unsigned char
// _Z32__spirv_SubgroupShuffleDownINTELIsET_S0_S0_j - short
// _Z32__spirv_SubgroupShuffleDownINTELItET_S0_S0_j - unsigned short
#define __AMDGCN_CLC_SUBGROUP_DOWN_TO_I32(TYPE, MANGLED_TYPE_NAME)             \
  _CLC_DEF TYPE                                                                \
      _Z32__spirv_SubgroupShuffleDownINTELI##MANGLED_TYPE_NAME##ET_S0_S0_j(    \
          TYPE var, TYPE lane_delta, unsigned int width) {                     \
    return _Z32__spirv_SubgroupShuffleDownINTELIiET_S0_S0_j(var, lane_delta,   \
                                                            width);            \
  }
__AMDGCN_CLC_SUBGROUP_DOWN_TO_I32(char, a);
__AMDGCN_CLC_SUBGROUP_DOWN_TO_I32(unsigned char, h);
__AMDGCN_CLC_SUBGROUP_DOWN_TO_I32(short, s);
__AMDGCN_CLC_SUBGROUP_DOWN_TO_I32(unsigned short, t);
#undef __AMDGCN_CLC_SUBGROUP_DOWN_TO_I32

// 32-bit types.
// __spirv_SubgroupShuffleDownINTEL - unsigned int
// __spirv_SubgroupShuffleDownINTEL - float
#define __AMDGCN_CLC_SUBGROUP_DOWN_I32(TYPE, CAST_TYPE, MANGLED_TYPE_NAME)     \
  _CLC_DEF TYPE                                                                \
      _Z32__spirv_SubgroupShuffleDownINTELI##MANGLED_TYPE_NAME##ET_S0_S0_j(    \
          TYPE var, TYPE lane_delta, unsigned int width) {                     \
    return __builtin_astype(_Z32__spirv_SubgroupShuffleDownINTELIiET_S0_S0_j(  \
                                as_int(var), as_int(lane_delta), width),       \
                            CAST_TYPE);                                        \
  }
__AMDGCN_CLC_SUBGROUP_DOWN_I32(unsigned int, uint, j);
__AMDGCN_CLC_SUBGROUP_DOWN_I32(float, float, f);
#undef __AMDGCN_CLC_SUBGROUP_DOWN_I32

// 64-bit types.
// double __spirv_SubgroupShuffleDownINTEL<double>(double, unsigned int, int)
#define __AMDGCN_CLC_SUBGROUP_DOWN_I64(TYPE, CAST_TYPE, MANGLED_TYPE_NAME)     \
  _CLC_DEF TYPE                                                                \
      _Z32__spirv_SubgroupShuffleDownINTELI##MANGLED_TYPE_NAME##ET_S0_S0_j(    \
          TYPE var, TYPE lane_delta, unsigned int width) {                     \
    int2 tmp = as_int2(var);                                                   \
    tmp.lo = _Z32__spirv_SubgroupShuffleDownINTELIiET_S0_S0_j(                 \
        tmp.lo, (int)lane_delta, width);                                       \
    tmp.hi = _Z32__spirv_SubgroupShuffleDownINTELIiET_S0_S0_j(                 \
        tmp.hi, (int)lane_delta, width);                                       \
    return __builtin_astype(tmp, CAST_TYPE);                                   \
  }
__AMDGCN_CLC_SUBGROUP_DOWN_I64(long, long, l);
__AMDGCN_CLC_SUBGROUP_DOWN_I64(unsigned long, ulong, m);
__AMDGCN_CLC_SUBGROUP_DOWN_I64(double, double, d);
#undef __AMDGCN_CLC_SUBGROUP_DOWN_I64

// Vector types.
#define __AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(TYPE, MANGLED_SCALAR_TY, NUM_ELEMS)               \
  _CLC_DEF TYPE                                                                             \
      _Z32__spirv_SubgroupShuffleDownINTELIDv##NUM_ELEMS##_##MANGLED_SCALAR_TY##ET_S1_S1_j( \
          TYPE var, TYPE lane_delta, unsigned int width) {                                  \
    TYPE res;                                                                               \
    for (int i = 0; i < NUM_ELEMS; ++i) {                                                   \
      res[i] =                                                                              \
          _Z32__spirv_SubgroupShuffleDownINTELI##MANGLED_SCALAR_TY##ET_S0_S0_j(             \
              var[i], (unsigned int)lane_delta[0], width);                                  \
    }                                                                                       \
    return res;                                                                             \
  }
// [u]char
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(char2, a, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(char4, a, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(char8, a, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(char16, a, 16)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(uchar2, h, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(uchar4, h, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(uchar8, h, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(uchar16, h, 16)
// [u]short
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(short2, s, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(short4, s, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(short8, s, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(short16, s, 16)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(ushort2, t, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(ushort4, t, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(ushort8, t, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(ushort16, t, 16)
// [u]int
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(int2, i, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(int4, i, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(int8, i, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(int16, i, 16)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(uint2, j, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(uint4, j, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(uint8, j, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(uint16, j, 16)
// [u]long
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(long2, l, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(long4, l, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(long8, l, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(long16, l, 16)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(ulong2, m, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(ulong4, m, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(ulong8, m, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(ulong16, m, 16)
// float
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(float2, f, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(float4, f, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(float8, f, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(float16, f, 16)
// double
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(double2, d, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(double4, d, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(double8, d, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(double16, d, 16)
#undef __AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC
