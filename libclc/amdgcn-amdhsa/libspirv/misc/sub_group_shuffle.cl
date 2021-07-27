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

// unsigned int __spirv_SubgroupShuffleINTEL<unsigned int>(unsigned int,
//                                                         unsigned int);
_CLC_DEF unsigned int
_Z28__spirv_SubgroupShuffleINTELIjET_S0_j(unsigned int Data,
                                          unsigned int InvocationId) {
  return as_uint(
      _Z28__spirv_SubgroupShuffleINTELIiET_S0_j(as_int(Data), InvocationId));
}
// Sub 32-bit types.
// _Z28__spirv_SubgroupShuffleINTELIaET_S0_j - char
// _Z28__spirv_SubgroupShuffleINTELIhET_S0_j - unsigned char
// _Z28__spirv_SubgroupShuffleINTELIsET_S0_j - long
// _Z28__spirv_SubgroupShuffleINTELItET_S0_j - unsigned long
#define __AMDGCN_CLC_SUBGROUP_TO_I32(TYPE, MANGLED_TYPE_NAME)                  \
  _CLC_DEF TYPE _Z28__spirv_SubgroupShuffleINTELI##MANGLED_TYPE_NAME##ET_S0_j( \
      TYPE Data, unsigned int InvocationId) {                                  \
    return _Z28__spirv_SubgroupShuffleINTELIiET_S0_j(Data, InvocationId);      \
  }
__AMDGCN_CLC_SUBGROUP_TO_I32(char, a);
__AMDGCN_CLC_SUBGROUP_TO_I32(unsigned char, h);
__AMDGCN_CLC_SUBGROUP_TO_I32(short, s);
__AMDGCN_CLC_SUBGROUP_TO_I32(unsigned short, t);
#undef __AMDGCN_CLC_SUBGROUP_TO_I32

// float __spirv_SubgroupShuffleINTEL<float>(float, unsigned int)
_CLC_DEF float
_Z28__spirv_SubgroupShuffleINTELIfET_S0_j(float Data,
                                          unsigned int InvocationId) {
  return as_float(
      _Z28__spirv_SubgroupShuffleINTELIiET_S0_j(as_int(Data), InvocationId));
}

// double __spirv_SubgroupShuffleINTEL<double>(double, unsigned int)
_CLC_DEF double
_Z28__spirv_SubgroupShuffleINTELIdET_S0_j(double Data,
                                          unsigned int InvocationId) {
  int2 tmp = as_int2(Data);
  tmp.lo = _Z28__spirv_SubgroupShuffleINTELIiET_S0_j(tmp.lo, InvocationId);
  tmp.hi = _Z28__spirv_SubgroupShuffleINTELIiET_S0_j(tmp.hi, InvocationId);
  return as_double(tmp);
}

// long __spirv_SubgroupShuffleINTEL<long>(long, unsigned int)
_CLC_DEF long
_Z28__spirv_SubgroupShuffleINTELIlET_S0_j(long Data,
                                          unsigned int InvocationId) {
  int2 tmp = as_int2(Data);
  tmp.lo = _Z28__spirv_SubgroupShuffleINTELIiET_S0_j(tmp.lo, InvocationId);
  tmp.hi = _Z28__spirv_SubgroupShuffleINTELIiET_S0_j(tmp.hi, InvocationId);
  return as_long(tmp);
}

// unsigned long __spirv_SubgroupShuffleINTEL<unsigned long>(unsigned long,
//                                                           unsigned int);
_CLC_DEF unsigned long
_Z28__spirv_SubgroupShuffleINTELImET_S0_j(unsigned long Data,
                                          unsigned int InvocationId) {
  int2 tmp = as_int2(Data);
  tmp.lo = _Z28__spirv_SubgroupShuffleINTELIjET_S0_j(tmp.lo, InvocationId);
  tmp.hi = _Z28__spirv_SubgroupShuffleINTELIjET_S0_j(tmp.hi, InvocationId);
  return as_ulong(tmp);
}

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

// unsigned int __spirv_SubgroupShuffleXorINTEL<unsigned int>(unsigned int,
//                                                            unsigned int);
_CLC_DEF unsigned int
_Z31__spirv_SubgroupShuffleXorINTELIjET_S0_j(unsigned int Data,
                                             unsigned int InvocationId) {
  return as_uint(
      _Z31__spirv_SubgroupShuffleXorINTELIiET_S0_j(as_int(Data), InvocationId));
}
// Sub 32-bit types.
// _Z31__spirv_SubgroupShuffleXorINTELIaET_S0_j - char
// _Z31__spirv_SubgroupShuffleXorINTELIhET_S0_j - unsigned char
// _Z31__spirv_SubgroupShuffleXorINTELIsET_S0_j - short
// _Z31__spirv_SubgroupShuffleXorINTELItET_S0_j - unsigned short
#define __AMDGCN_CLC_SUBGROUP_XOR_TO_I32(TYPE, MANGLED_TYPE_NAME)              \
  _CLC_DEF TYPE                                                                \
      _Z31__spirv_SubgroupShuffleXorINTELI##MANGLED_TYPE_NAME##ET_S0_j(        \
          TYPE Data, unsigned int InvocationId) {                              \
    return _Z31__spirv_SubgroupShuffleXorINTELIiET_S0_j(Data, InvocationId);   \
  }
__AMDGCN_CLC_SUBGROUP_XOR_TO_I32(char, a);
__AMDGCN_CLC_SUBGROUP_XOR_TO_I32(unsigned char, h);
__AMDGCN_CLC_SUBGROUP_XOR_TO_I32(short, s);
__AMDGCN_CLC_SUBGROUP_XOR_TO_I32(unsigned short, t);
#undef __AMDGCN_CLC_SUBGROUP_XOR_TO_I32

// float __spirv_SubgroupShuffleXorINTEL<float>(float, unsigned int)
_CLC_DEF float
_Z31__spirv_SubgroupShuffleXorINTELIfET_S0_j(float Data,
                                             unsigned int InvocationId) {
  return as_float(
      _Z31__spirv_SubgroupShuffleXorINTELIiET_S0_j(as_int(Data), InvocationId));
}

// double __spirv_SubgroupShuffleXorINTEL<double>(double, unsigned int)
_CLC_DEF double
_Z31__spirv_SubgroupShuffleXorINTELIdET_S0_j(double Data,
                                             unsigned int InvocationId) {
  int2 tmp = as_int2(Data);
  tmp.lo = _Z31__spirv_SubgroupShuffleXorINTELIiET_S0_j(tmp.lo, InvocationId);
  tmp.hi = _Z31__spirv_SubgroupShuffleXorINTELIiET_S0_j(tmp.hi, InvocationId);
  return as_double(tmp);
}

// long __spirv_SubgroupShuffleXorINTEL<long>(long, unsigned int)
_CLC_DEF long
_Z31__spirv_SubgroupShuffleXorINTELIlET_S0_j(long Data,
                                             unsigned int InvocationId) {
  int2 tmp = as_int2(Data);
  tmp.lo = _Z31__spirv_SubgroupShuffleXorINTELIiET_S0_j(tmp.lo, InvocationId);
  tmp.hi = _Z31__spirv_SubgroupShuffleXorINTELIiET_S0_j(tmp.hi, InvocationId);
  return as_long(tmp);
}

// unsigned long __spirv_SubgroupShuffleXorINTEL<unsigned long>(unsigned long,
//                                                              unsigned int);
_CLC_DEF unsigned long
_Z31__spirv_SubgroupShuffleXorINTELImET_S0_j(unsigned long Data,
                                             unsigned int InvocationId) {
  uint2 tmp = as_uint2(Data);
  tmp.lo = _Z31__spirv_SubgroupShuffleXorINTELIjET_S0_j(tmp.lo, InvocationId);
  tmp.hi = _Z31__spirv_SubgroupShuffleXorINTELIjET_S0_j(tmp.hi, InvocationId);
  return as_ulong(tmp);
}

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

// unsigned int __spirv_SubgroupShuffleUpINTEL<unsigned int>(unsigned int,
//                                                           unisgned int,
//                                                           unsigned int);
_CLC_DEF unsigned int _Z30__spirv_SubgroupShuffleUpINTELIjET_S0_S0_j(
    unsigned int var, unsigned int lane_delta, unsigned int width) {
  return as_uint(_Z30__spirv_SubgroupShuffleUpINTELIiET_S0_S0_j(
      as_int(var), as_int(lane_delta), width));
}
// Sub 32-bit types.
// _Z30__spirv_SubgroupShuffleUpINTELIaET_S0_S0_j - char
// _Z30__spirv_SubgroupShuffleUpINTELIhET_S0_S0_j - unsigned char
// _Z30__spirv_SubgroupShuffleUpINTELIsET_S0_S0_j - short
// _Z30__spirv_SubgroupShuffleUpINTELItET_S0_S0_j - unsigned short
#define __AMDGCN_CLC_SUBGROUP_UP_TO_I32(TYPE, MANGLED_TYPE_NAME)               \
  _CLC_DEF TYPE                                                                \
      _Z30__spirv_SubgroupShuffleUpINTELI##MANGLED_TYPE_NAME##ET_S0_S0_j(      \
          TYPE var, TYPE lane_delta, unsigned int width) {                     \
    return _Z30__spirv_SubgroupShuffleUpINTELIiET_S0_S0_j(var, lane_delta,     \
                                                          width);              \
  }
__AMDGCN_CLC_SUBGROUP_UP_TO_I32(char, a);
__AMDGCN_CLC_SUBGROUP_UP_TO_I32(unsigned char, h);
__AMDGCN_CLC_SUBGROUP_UP_TO_I32(short, s);
__AMDGCN_CLC_SUBGROUP_UP_TO_I32(unsigned short, t);
#undef __AMDGCN_CLC_SUBGROUP_UP_TO_I32

// float __spirv_SubgroupShuffleUpINTEL<float>(float,
//                                             float,
//                                             unsigned int)
_CLC_DEF float
_Z30__spirv_SubgroupShuffleUpINTELIfET_S0_S0_j(float var, float lane_delta,
                                               unsigned int width) {
  return as_float(_Z30__spirv_SubgroupShuffleUpINTELIiET_S0_S0_j(
      as_int(var), as_int(lane_delta), width));
}

// double __spirv_SubgroupShuffleUpINTEL<double>(double,
//                                               double,
//                                               unsigned int)
_CLC_DEF double
_Z30__spirv_SubgroupShuffleUpINTELIdET_S0_S0_j(double var, double lane_delta,
                                               unsigned int width) {
  int2 tmp = as_int2(var);
  tmp.lo = _Z30__spirv_SubgroupShuffleUpINTELIiET_S0_S0_j(
      tmp.lo, (int)lane_delta, width);
  tmp.hi = _Z30__spirv_SubgroupShuffleUpINTELIiET_S0_S0_j(
      tmp.hi, (int)lane_delta, width);
  return as_double(tmp);
}

// long __spirv_SubgroupShuffleUpINTEL<long>(long, long, unsigned int)
_CLC_DEF long
_Z30__spirv_SubgroupShuffleUpINTELIlET_S0_S0_j(long var, long lane_delta,
                                               unsigned int width) {
  int2 tmp = as_int2(var);
  tmp.lo = _Z30__spirv_SubgroupShuffleUpINTELIiET_S0_S0_j(
      tmp.lo, (int)lane_delta, width);
  tmp.hi = _Z30__spirv_SubgroupShuffleUpINTELIiET_S0_S0_j(
      tmp.hi, (int)lane_delta, width);
  return as_long(tmp);
}

// unsigned long __spirv_SubgroupShuffleUpINTEL<unsigned long>(unsigned long,
//                                                             unsigned long,
//                                                             unsigned int);
_CLC_DEF unsigned long _Z30__spirv_SubgroupShuffleUpINTELImET_S0_S0_j(
    unsigned long var, unsigned long lane_delta, unsigned int width) {
  uint2 tmp = as_uint2(var);
  tmp.lo = _Z30__spirv_SubgroupShuffleUpINTELIjET_S0_S0_j(
      tmp.lo, (unsigned int)lane_delta, width);
  tmp.hi = _Z30__spirv_SubgroupShuffleUpINTELIjET_S0_S0_j(
      tmp.hi, (unsigned int)lane_delta, width);
  return as_ulong(tmp);
}

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

// unsigned int __spirv_SubgroupShuffleDownINTEL<unsigned int>(unsigned int,
//                                                             unsigned int,
//                                                             unsigned int);
_CLC_DEF unsigned int _Z32__spirv_SubgroupShuffleDownINTELIjET_S0_S0_j(
    unsigned int var, unsigned int lane_delta, unsigned int width) {
  return as_uint(_Z32__spirv_SubgroupShuffleDownINTELIiET_S0_S0_j(
      as_int(var), as_int(lane_delta), width));
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

// float __spirv_SubgroupShuffleDownINTEL<float>(float, float, int)
_CLC_DEF float
_Z32__spirv_SubgroupShuffleDownINTELIfET_S0_S0_j(float var, float lane_delta,
                                                 unsigned int width) {
  return as_float(_Z32__spirv_SubgroupShuffleDownINTELIiET_S0_S0_j(
      as_int(var), as_int(lane_delta), width));
}

// double __spirv_SubgroupShuffleDownINTEL<double>(double, unsigned int, int)
_CLC_DEF double
_Z32__spirv_SubgroupShuffleDownINTELIdET_S0_S0_j(double var, double lane_delta,
                                                 unsigned int width) {
  int2 tmp = as_int2(var);
  tmp.lo = _Z32__spirv_SubgroupShuffleDownINTELIiET_S0_S0_j(
      tmp.lo, (int)lane_delta, width);
  tmp.hi = _Z32__spirv_SubgroupShuffleDownINTELIiET_S0_S0_j(
      tmp.hi, (int)lane_delta, width);
  return as_double(tmp);
}

// long __spirv_SubgroupShuffleDownINTEL<long>(long, long, int)
_CLC_DEF long
_Z32__spirv_SubgroupShuffleDownINTELIlET_S0_S0_j(long var, long lane_delta,
                                                 unsigned int width) {
  int2 tmp = as_int2(var);
  tmp.lo = _Z32__spirv_SubgroupShuffleDownINTELIiET_S0_S0_j(
      tmp.lo, (int)lane_delta, width);
  tmp.hi = _Z32__spirv_SubgroupShuffleDownINTELIiET_S0_S0_j(
      tmp.hi, (int)lane_delta, width);
  return as_long(tmp);
}

// unsigned long __spirv_SubgroupShuffleDownINTEL<unsigned long>(unsigned long,
//                                                               unsigned long,
//                                                               int);
_CLC_DEF unsigned long _Z32__spirv_SubgroupShuffleDownINTELImET_S0_S0_j(
    unsigned long var, unsigned long lane_delta, unsigned int width) {
  uint2 tmp = as_uint2(var);
  tmp.lo = _Z32__spirv_SubgroupShuffleDownINTELIjET_S0_S0_j(
      tmp.lo, (unsigned int)lane_delta, width);
  tmp.hi = _Z32__spirv_SubgroupShuffleDownINTELIjET_S0_S0_j(
      tmp.hi, (unsigned int)lane_delta, width);
  return as_ulong(tmp);
}

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
