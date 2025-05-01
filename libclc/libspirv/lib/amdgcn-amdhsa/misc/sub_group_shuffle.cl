//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

#define SELF __spirv_SubgroupLocalInvocationId();
#define SUBGROUP_SIZE __spirv_SubgroupMaxSize()

// Shuffle
_CLC_OVERLOAD _CLC_DEF int
__spirv_SubgroupShuffleINTEL(int Data, unsigned int InvocationId) {
  int Index = InvocationId;
  return __builtin_amdgcn_ds_bpermute(Index << 2, Data);
}
// int __spirv_SubgroupShuffleINTEL<int>(int, unsigned int)
_CLC_DEF int
_Z28__spirv_SubgroupShuffleINTELIiET_S0_j(int Data, unsigned int InvocationId) {
  return __spirv_SubgroupShuffleINTEL(Data, InvocationId);
}

// Sub 32-bit types.
#define __AMDGCN_CLC_SUBGROUP_SUB_I32(TYPE)                                    \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleINTEL(                    \
      TYPE Data, unsigned int InvocationId) {                                  \
    return __spirv_SubgroupShuffleINTEL((int)Data, InvocationId);              \
  }
__AMDGCN_CLC_SUBGROUP_SUB_I32(char);
__AMDGCN_CLC_SUBGROUP_SUB_I32(unsigned char);
__AMDGCN_CLC_SUBGROUP_SUB_I32(short);
__AMDGCN_CLC_SUBGROUP_SUB_I32(unsigned short);
#undef __AMDGCN_CLC_SUBGROUP_SUB_I32

// _Z28__spirv_SubgroupShuffleINTELIaET_S0_j - char
// _Z28__spirv_SubgroupShuffleINTELIhET_S0_j - unsigned char
// _Z28__spirv_SubgroupShuffleINTELIsET_S0_j - long
// _Z28__spirv_SubgroupShuffleINTELItET_S0_j - unsigned long
// _Z28__spirv_SubgroupShuffleINTELIDhET_S0_j - half
#define __AMDGCN_CLC_SUBGROUP_SUB_I32(TYPE, MANGLED_TYPE_NAME)                 \
  _CLC_DEF TYPE _Z28__spirv_SubgroupShuffleINTELI##MANGLED_TYPE_NAME##ET_S0_j( \
      TYPE Data, unsigned int InvocationId) {                                  \
    return __spirv_SubgroupShuffleINTEL(Data, InvocationId);                   \
  }
__AMDGCN_CLC_SUBGROUP_SUB_I32(char, a);
__AMDGCN_CLC_SUBGROUP_SUB_I32(unsigned char, h);
__AMDGCN_CLC_SUBGROUP_SUB_I32(short, s);
__AMDGCN_CLC_SUBGROUP_SUB_I32(unsigned short, t);

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
_CLC_OVERLOAD _CLC_DEF half
__spirv_SubgroupShuffleINTEL(half Data, unsigned int InvocationId) {
  unsigned short tmp = __clc_as_ushort(Data);
  tmp = __spirv_SubgroupShuffleINTEL(tmp, InvocationId);
  return __clc_as_half(tmp);
}
_CLC_DEF half _Z28__spirv_SubgroupShuffleINTELIDhET_S0_j(
    half Data, unsigned int InvocationId) {
  return __spirv_SubgroupShuffleINTEL(Data, InvocationId);
}
#endif // cl_khr_fp16

#undef __AMDGCN_CLC_SUBGROUP_SUB_I32

// 32-bit types.
#define __AMDGCN_CLC_SUBGROUP_I32(TYPE, CAST_TYPE)                             \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleINTEL(                    \
      TYPE Data, unsigned int InvocationId) {                                  \
    return __builtin_astype(                                                   \
        __spirv_SubgroupShuffleINTEL(__clc_as_int(Data), InvocationId),        \
        CAST_TYPE);                                                            \
  }
__AMDGCN_CLC_SUBGROUP_I32(unsigned int, uint);
__AMDGCN_CLC_SUBGROUP_I32(float, float);
#undef __AMDGCN_CLC_SUBGROUP_I32

// __spirv_SubgroupShuffleINTEL - unsigned int
// __spirv_SubgroupShuffleINTEL-  float
#define __AMDGCN_CLC_SUBGROUP_I32(TYPE, CAST_TYPE, MANGLED_TYPE_NAME)          \
  _CLC_DEF TYPE _Z28__spirv_SubgroupShuffleINTELI##MANGLED_TYPE_NAME##ET_S0_j( \
      TYPE Data, unsigned int InvocationId) {                                  \
    return __spirv_SubgroupShuffleINTEL(Data, InvocationId);                   \
  }
__AMDGCN_CLC_SUBGROUP_I32(unsigned int, uint, j);
__AMDGCN_CLC_SUBGROUP_I32(float, float, f);
#undef __AMDGCN_CLC_SUBGROUP_I32

// 64-bit types.
#define __AMDGCN_CLC_SUBGROUP_I64(TYPE, CAST_TYPE)                             \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleINTEL(                    \
      TYPE Data, unsigned int InvocationId) {                                  \
    int2 tmp = __clc_as_int2(Data);                                            \
    tmp.lo = __spirv_SubgroupShuffleINTEL(tmp.lo, InvocationId);               \
    tmp.hi = __spirv_SubgroupShuffleINTEL(tmp.hi, InvocationId);               \
    return __builtin_astype(tmp, CAST_TYPE);                                   \
  }
__AMDGCN_CLC_SUBGROUP_I64(long, long);
__AMDGCN_CLC_SUBGROUP_I64(unsigned long, ulong);
__AMDGCN_CLC_SUBGROUP_I64(double, double);
#undef __AMDGCN_CLC_SUBGROUP_I64

// __spirv_SubgroupShuffleINTEL - long
// __spirv_SubgroupShuffleINTEL - unsigned long
// __spirv_SubgroupShuffleINTEL - double
#define __AMDGCN_CLC_SUBGROUP_I64(TYPE, CAST_TYPE, MANGLED_TYPE_NAME)          \
  _CLC_DEF TYPE _Z28__spirv_SubgroupShuffleINTELI##MANGLED_TYPE_NAME##ET_S0_j( \
      TYPE Data, unsigned int InvocationId) {                                  \
    return __spirv_SubgroupShuffleINTEL(Data, InvocationId);                   \
  }
__AMDGCN_CLC_SUBGROUP_I64(long, long, l);
__AMDGCN_CLC_SUBGROUP_I64(unsigned long, ulong, m);
__AMDGCN_CLC_SUBGROUP_I64(double, double, d);
#undef __AMDGCN_CLC_SUBGROUP_I64

// Vector types.
#define __AMDGCN_CLC_SUBGROUP_TO_VEC(TYPE, NUM_ELEMS)                          \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleINTEL(                    \
      TYPE Data, unsigned int InvocationId) {                                  \
    TYPE res;                                                                  \
    for (int i = 0; i < NUM_ELEMS; ++i) {                                      \
      res[i] = __spirv_SubgroupShuffleINTEL(Data[i], InvocationId);            \
    }                                                                          \
    return res;                                                                \
  }

// [u]char
__AMDGCN_CLC_SUBGROUP_TO_VEC(char2, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(char4, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(char8, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(char16, 16)
__AMDGCN_CLC_SUBGROUP_TO_VEC(uchar2, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(uchar4, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(uchar8, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(uchar16, 16)
// [u]short
__AMDGCN_CLC_SUBGROUP_TO_VEC(short2, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(short4, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(short8, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(short16, 16)
__AMDGCN_CLC_SUBGROUP_TO_VEC(ushort2, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(ushort4, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(ushort8, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(ushort16, 16)
// [u]int
__AMDGCN_CLC_SUBGROUP_TO_VEC(int2, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(int4, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(int8, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(int16, 16)
__AMDGCN_CLC_SUBGROUP_TO_VEC(uint2, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(uint4, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(uint8, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(uint16, 16)
// [u]long
__AMDGCN_CLC_SUBGROUP_TO_VEC(long2, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(long4, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(long8, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(long16, 16)
__AMDGCN_CLC_SUBGROUP_TO_VEC(ulong2, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(ulong4, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(ulong8, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(ulong16, 16)
// half
#ifdef cl_khr_fp16
__AMDGCN_CLC_SUBGROUP_TO_VEC(half2, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(half4, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(half8, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(half16, 16)
#endif // cl_khr_fp16
// float
__AMDGCN_CLC_SUBGROUP_TO_VEC(float2, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(float4, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(float8, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(float16, 16)
// double
__AMDGCN_CLC_SUBGROUP_TO_VEC(double2, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(double4, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(double8, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(double16, 16)
#undef __AMDGCN_CLC_SUBGROUP_TO_VEC

#define __AMDGCN_CLC_SUBGROUP_TO_VEC(TYPE, MANGLED_SCALAR_TY, NUM_ELEMS)             \
  _CLC_DEF TYPE                                                                      \
      _Z28__spirv_SubgroupShuffleINTELIDv##NUM_ELEMS##_##MANGLED_SCALAR_TY##ET_S1_j( \
          TYPE Data, unsigned int InvocationId) {                                    \
    return __spirv_SubgroupShuffleINTEL(Data, InvocationId);                         \
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
// half
#ifdef cl_khr_fp16
__AMDGCN_CLC_SUBGROUP_TO_VEC(half2, Dh, 2)
__AMDGCN_CLC_SUBGROUP_TO_VEC(half4, Dh, 4)
__AMDGCN_CLC_SUBGROUP_TO_VEC(half8, Dh, 8)
__AMDGCN_CLC_SUBGROUP_TO_VEC(half16, Dh, 16)
#endif // cl_khr_fp16
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
_CLC_OVERLOAD _CLC_DEF int
__spirv_SubgroupShuffleXorINTEL(int Data, unsigned int InvocationId) {
  int self = SELF;
  unsigned int index = self ^ InvocationId;
  index =
      index >= ((self + SUBGROUP_SIZE) & ~(SUBGROUP_SIZE - 1)) ? self : index;
  return __builtin_amdgcn_ds_bpermute(index << 2, Data);
}
// int __spirv_SubgroupShuffleXorINTEL<int>(int, unsigned int)
_CLC_DEF int
_Z31__spirv_SubgroupShuffleXorINTELIiET_S0_j(int Data,
                                             unsigned int InvocationId) {
  return __spirv_SubgroupShuffleXorINTEL(Data, InvocationId);
}

// Sub 32-bit types.
#define __AMDGCN_CLC_SUBGROUP_XOR_SUB_I32(TYPE)                                \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleXorINTEL(                 \
      TYPE Data, unsigned int InvocationId) {                                  \
    return __spirv_SubgroupShuffleXorINTEL((int)Data, InvocationId);           \
  }
__AMDGCN_CLC_SUBGROUP_XOR_SUB_I32(char);
__AMDGCN_CLC_SUBGROUP_XOR_SUB_I32(unsigned char);
__AMDGCN_CLC_SUBGROUP_XOR_SUB_I32(short);
__AMDGCN_CLC_SUBGROUP_XOR_SUB_I32(unsigned short);
#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DEF half __spirv_SubgroupShuffleXorINTEL(
    half Data, unsigned int InvocationId) {
  unsigned short tmp = __clc_as_ushort(Data);
  tmp = (unsigned short)__spirv_SubgroupShuffleXorINTEL(tmp, InvocationId);
  return __clc_as_half(tmp);
}
#endif // cl_khr_fp16
#undef __AMDGCN_CLC_SUBGROUP_XOR_SUB_I32

// _Z31__spirv_SubgroupShuffleXorINTELIaET_S0_j - char
// _Z31__spirv_SubgroupShuffleXorINTELIhET_S0_j - unsigned char
// _Z31__spirv_SubgroupShuffleXorINTELIsET_S0_j - short
// _Z31__spirv_SubgroupShuffleXorINTELItET_S0_j - unsigned short
// _Z31__spirv_SubgroupShuffleXorINTELIDhET_S0_j - half
#define __AMDGCN_CLC_SUBGROUP_XOR_SUB_I32(TYPE, MANGLED_TYPE_NAME)             \
  _CLC_DEF TYPE                                                                \
      _Z31__spirv_SubgroupShuffleXorINTELI##MANGLED_TYPE_NAME##ET_S0_j(        \
          TYPE Data, unsigned int InvocationId) {                              \
    return __spirv_SubgroupShuffleXorINTEL(Data, InvocationId);                \
  }
__AMDGCN_CLC_SUBGROUP_XOR_SUB_I32(char, a);
__AMDGCN_CLC_SUBGROUP_XOR_SUB_I32(unsigned char, h);
__AMDGCN_CLC_SUBGROUP_XOR_SUB_I32(short, s);
__AMDGCN_CLC_SUBGROUP_XOR_SUB_I32(unsigned short, t);
#ifdef cl_khr_fp16
_CLC_DEF half _Z31__spirv_SubgroupShuffleXorINTELIDhET_S0_j(
    half Data, unsigned int InvocationId) {
  return __spirv_SubgroupShuffleXorINTEL(Data, InvocationId);
}
#endif // cl_khr_fp16
#undef __AMDGCN_CLC_SUBGROUP_XOR_SUB_I32

// 32-bit types.
#define __AMDGCN_CLC_SUBGROUP_XOR_I32(TYPE, CAST_TYPE)                         \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleXorINTEL(                 \
      TYPE Data, unsigned int InvocationId) {                                  \
    return __builtin_astype(                                                   \
        __spirv_SubgroupShuffleXorINTEL(__clc_as_int(Data), InvocationId),     \
        CAST_TYPE);                                                            \
  }
__AMDGCN_CLC_SUBGROUP_XOR_I32(unsigned int, uint);
__AMDGCN_CLC_SUBGROUP_XOR_I32(float, float);
#undef __AMDGCN_CLC_SUBGROUP_XOR_I32

// __spirv_SubgroupShuffleXorINTEL - unsigned int
// __spirv_SubgroupShuffleXorINTEL - float
#define __AMDGCN_CLC_SUBGROUP_XOR_I32(TYPE, CAST_TYPE, MANGLED_TYPE_NAME)      \
  _CLC_DEF TYPE                                                                \
      _Z31__spirv_SubgroupShuffleXorINTELI##MANGLED_TYPE_NAME##ET_S0_j(        \
          TYPE Data, unsigned int InvocationId) {                              \
    return __spirv_SubgroupShuffleXorINTEL(Data, InvocationId);                \
  }
__AMDGCN_CLC_SUBGROUP_XOR_I32(unsigned int, uint, j);
__AMDGCN_CLC_SUBGROUP_XOR_I32(float, float, f);
#undef __AMDGCN_CLC_SUBGROUP_XOR_I32

// 64-bit types.
#define __AMDGCN_CLC_SUBGROUP_XOR_I64(TYPE, CAST_TYPE)                         \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleXorINTEL(                 \
      TYPE Data, unsigned int InvocationId) {                                  \
    int2 tmp = __clc_as_int2(Data);                                            \
    tmp.lo = __spirv_SubgroupShuffleXorINTEL(tmp.lo, InvocationId);            \
    tmp.hi = __spirv_SubgroupShuffleXorINTEL(tmp.hi, InvocationId);            \
    return __builtin_astype(tmp, CAST_TYPE);                                   \
  }
__AMDGCN_CLC_SUBGROUP_XOR_I64(long, long);
__AMDGCN_CLC_SUBGROUP_XOR_I64(unsigned long, ulong);
__AMDGCN_CLC_SUBGROUP_XOR_I64(double, double);
#undef __AMDGCN_CLC_SUBGROUP_XOR_I64

// __spirv_SubgroupShuffleXorINTEL - long
// __spirv_SubgroupShuffleXorINTEL - unsigned long
// __spirv_SubgroupShuffleXorINTEL - double
#define __AMDGCN_CLC_SUBGROUP_XOR_I64(TYPE, CAST_TYPE, MANGLED_TYPE_NAME)      \
  _CLC_DEF TYPE                                                                \
      _Z31__spirv_SubgroupShuffleXorINTELI##MANGLED_TYPE_NAME##ET_S0_j(        \
          TYPE Data, unsigned int InvocationId) {                              \
    return __spirv_SubgroupShuffleXorINTEL(Data, InvocationId);                \
  }
__AMDGCN_CLC_SUBGROUP_XOR_I64(long, long, l);
__AMDGCN_CLC_SUBGROUP_XOR_I64(unsigned long, ulong, m);
__AMDGCN_CLC_SUBGROUP_XOR_I64(double, double, d);
#undef __AMDGCN_CLC_SUBGROUP_XOR_I64

// Vector types.

#define __AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(TYPE, NUM_ELEMS)                      \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleXorINTEL(                 \
      TYPE Data, unsigned int InvocationId) {                                  \
    TYPE res;                                                                  \
    for (int i = 0; i < NUM_ELEMS; ++i) {                                      \
      res[i] = __spirv_SubgroupShuffleXorINTEL(Data[i], InvocationId);         \
    }                                                                          \
    return res;                                                                \
  }
// [u]char
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(char2, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(char4, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(char8, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(char16, 16)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(uchar2, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(uchar4, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(uchar8, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(uchar16, 16)
// [u]short
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(short2, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(short4, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(short8, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(short16, 16)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(ushort2, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(ushort4, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(ushort8, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(ushort16, 16)
// [u]int
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(int2, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(int4, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(int8, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(int16, 16)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(uint2, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(uint4, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(uint8, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(uint16, 16)
// [u]long
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(long2, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(long4, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(long8, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(long16, 16)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(ulong2, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(ulong4, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(ulong8, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(ulong16, 16)
// float
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(float2, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(float4, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(float8, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(float16, 16)
// half
#ifdef cl_khr_fp16
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(half2, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(half4, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(half8, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(half16, 16)
#endif // cl_khr_fp16
// double
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(double2, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(double4, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(double8, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(double16, 16)
#undef __AMDGCN_CLC_SUBGROUP_XOR_TO_VEC

#define __AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(TYPE, MANGLED_SCALAR_TY, NUM_ELEMS)            \
  _CLC_DEF TYPE                                                                         \
      _Z31__spirv_SubgroupShuffleXorINTELIDv##NUM_ELEMS##_##MANGLED_SCALAR_TY##ET_S1_j( \
          TYPE Data, unsigned int InvocationId) {                                       \
    return __spirv_SubgroupShuffleXorINTEL(Data, InvocationId);                         \
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
// half
#ifdef cl_khr_fp16
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(half2, Dh, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(half4, Dh, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(half8, Dh, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(half16, Dh, 16)
#endif // cl_khr_fp16
// double
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(double2, d, 2)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(double4, d, 4)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(double8, d, 8)
__AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(double16, d, 16)
#undef __AMDGCN_CLC_SUBGROUP_XOR_TO_VEC

// Shuffle Up
_CLC_OVERLOAD _CLC_DEF int
__spirv_SubgroupShuffleUpINTEL(int previous, int current, unsigned int delta) {
  int self = SELF;
  int size = SUBGROUP_SIZE;

  int index = self - delta;

  int val;
  if (index >= 0 && index < size) {
    val = current;
  } else if (index < 0 && index > -size) {
    val = previous;
    index = index + size;
  } else {
    // index out of bounds so return arbitrary data
    val = current;
    index = self;
  }

  return __builtin_amdgcn_ds_bpermute(index << 2, val);
}
// int __spirv_SubgroupShuffleUpINTEL<int>(int, int, unsigned int)
_CLC_DEF int
_Z30__spirv_SubgroupShuffleUpINTELIiET_S0_S0_j(int previous, int current,
                                               unsigned int delta) {
  return __spirv_SubgroupShuffleUpINTEL(previous, current, delta);
}

// Sub 32-bit types.
#define __AMDGCN_CLC_SUBGROUP_UP_SUB_I32(TYPE)                                 \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleUpINTEL(                  \
      TYPE previous, TYPE current, unsigned int delta) {                       \
    return __spirv_SubgroupShuffleUpINTEL((int)previous, (int)current, delta); \
  }
__AMDGCN_CLC_SUBGROUP_UP_SUB_I32(char);
__AMDGCN_CLC_SUBGROUP_UP_SUB_I32(unsigned char);
__AMDGCN_CLC_SUBGROUP_UP_SUB_I32(short);
__AMDGCN_CLC_SUBGROUP_UP_SUB_I32(unsigned short);

// half
#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DEF half __spirv_SubgroupShuffleUpINTEL(
    half previous, half current, unsigned int delta) {
  unsigned short tmpP = __clc_as_ushort(previous);
  unsigned short tmpC = __clc_as_ushort(current);
  tmpC = __spirv_SubgroupShuffleUpINTEL(tmpP, tmpC, delta);
  return __clc_as_half(tmpC);
}
#endif // cl_khr_fp16
#undef __AMDGCN_CLC_SUBGROUP_UP_SUB_I32

// _Z30__spirv_SubgroupShuffleUpINTELIaET_S0_S0_j - char
// _Z30__spirv_SubgroupShuffleUpINTELIhET_S0_S0_j - unsigned char
// _Z30__spirv_SubgroupShuffleUpINTELIsET_S0_S0_j - short
// _Z30__spirv_SubgroupShuffleUpINTELItET_S0_S0_j - unsigned short
// _Z30__spirv_SubgroupShuffleUpINTELIDhET_S0_S0_j - half
#define __AMDGCN_CLC_SUBGROUP_UP_SUB_I32(TYPE, MANGLED_TYPE_NAME)              \
  _CLC_DEF TYPE                                                                \
      _Z30__spirv_SubgroupShuffleUpINTELI##MANGLED_TYPE_NAME##ET_S0_S0_j(      \
          TYPE previous, TYPE current, unsigned int delta) {                   \
    return __spirv_SubgroupShuffleUpINTEL(previous, current, delta);           \
  }
__AMDGCN_CLC_SUBGROUP_UP_SUB_I32(char, a);
__AMDGCN_CLC_SUBGROUP_UP_SUB_I32(unsigned char, h);
__AMDGCN_CLC_SUBGROUP_UP_SUB_I32(short, s);
__AMDGCN_CLC_SUBGROUP_UP_SUB_I32(unsigned short, t);
// half
#ifdef cl_khr_fp16
_CLC_DEF half _Z30__spirv_SubgroupShuffleUpINTELIDhET_S0_S0_j(
    half previous, half current, unsigned int delta) {
  return __spirv_SubgroupShuffleUpINTEL(previous, current, delta);
}
#endif // cl_khr_fp16
#undef __AMDGCN_CLC_SUBGROUP_UP_SUB_I32

// 32-bit types.
#define __AMDGCN_CLC_SUBGROUP_UP_I32(TYPE, CAST_TYPE)                          \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleUpINTEL(                  \
      TYPE previous, TYPE current, unsigned int delta) {                       \
    return __builtin_astype(                                                   \
        __spirv_SubgroupShuffleUpINTEL(__clc_as_int(previous),                 \
                                       __clc_as_int(current), delta),          \
        CAST_TYPE);                                                            \
  }
__AMDGCN_CLC_SUBGROUP_UP_I32(unsigned int, uint);
__AMDGCN_CLC_SUBGROUP_UP_I32(float, float);
#undef __AMDGCN_CLC_SUBGROUP_UP_I32

// __spirv_SubgroupShuffleUpINTELi - unsigned int
// __spirv_SubgroupShuffleUpINTELi - float
#define __AMDGCN_CLC_SUBGROUP_UP_I32(TYPE, CAST_TYPE, MANGLED_TYPE_NAME)       \
  _CLC_DEF TYPE                                                                \
      _Z30__spirv_SubgroupShuffleUpINTELI##MANGLED_TYPE_NAME##ET_S0_S0_j(      \
          TYPE previous, TYPE current, unsigned int delta) {                   \
    return __spirv_SubgroupShuffleUpINTEL(previous, current, delta);           \
  }
__AMDGCN_CLC_SUBGROUP_UP_I32(unsigned int, uint, j);
__AMDGCN_CLC_SUBGROUP_UP_I32(float, float, f);
#undef __AMDGCN_CLC_SUBGROUP_UP_I32

// 64-bit types.
#define __AMDGCN_CLC_SUBGROUP_UP_I64(TYPE, CAST_TYPE)                          \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleUpINTEL(                  \
      TYPE previous, TYPE current, unsigned int delta) {                       \
    int2 tmp_previous = __clc_as_int2(previous);                               \
    int2 tmp_current = __clc_as_int2(current);                                 \
    int2 ret;                                                                  \
    ret.lo = __spirv_SubgroupShuffleUpINTEL(tmp_previous.lo, tmp_current.lo,   \
                                            delta);                            \
    ret.hi = __spirv_SubgroupShuffleUpINTEL(tmp_previous.hi, tmp_current.hi,   \
                                            delta);                            \
    return __builtin_astype(ret, CAST_TYPE);                                   \
  }
__AMDGCN_CLC_SUBGROUP_UP_I64(long, long);
__AMDGCN_CLC_SUBGROUP_UP_I64(unsigned long, ulong);
__AMDGCN_CLC_SUBGROUP_UP_I64(double, double);
#undef __AMDGCN_CLC_SUBGROUP_UP_I64

// __spirv_SubgroupShuffleUpINTEL - long
// __spirv_SubgroupShuffleUpINTEL - unsigned long
// __spirv_SubgroupShuffleUpINTEL - double
#define __AMDGCN_CLC_SUBGROUP_UP_I64(TYPE, CAST_TYPE, MANGLED_TYPE_NAME)       \
  _CLC_DEF TYPE                                                                \
      _Z30__spirv_SubgroupShuffleUpINTELI##MANGLED_TYPE_NAME##ET_S0_S0_j(      \
          TYPE previous, TYPE current, unsigned int delta) {                   \
    return __spirv_SubgroupShuffleUpINTEL(previous, current, delta);           \
  }
__AMDGCN_CLC_SUBGROUP_UP_I64(long, long, l);
__AMDGCN_CLC_SUBGROUP_UP_I64(unsigned long, ulong, m);
__AMDGCN_CLC_SUBGROUP_UP_I64(double, double, d);
#undef __AMDGCN_CLC_SUBGROUP_UP_I64

// Vector types.
#define __AMDGCN_CLC_SUBGROUP_UP_TO_VEC(TYPE, NUM_ELEMS)                       \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleUpINTEL(                  \
      TYPE previous, TYPE current, unsigned int delta) {                       \
    TYPE res;                                                                  \
    for (int i = 0; i < NUM_ELEMS; ++i) {                                      \
      res[i] = __spirv_SubgroupShuffleUpINTEL(previous[i], current[i], delta); \
    }                                                                          \
    return res;                                                                \
  }
// [u]char
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(char2, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(char4, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(char8, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(char16, 16)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(uchar2, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(uchar4, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(uchar8, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(uchar16, 16)
// [u]short
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(short2, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(short4, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(short8, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(short16, 16)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(ushort2, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(ushort4, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(ushort8, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(ushort16, 16)
// [u]int
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(int2, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(int4, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(int8, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(int16, 16)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(uint2, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(uint4, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(uint8, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(uint16, 16)
// [u]long
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(long2, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(long4, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(long8, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(long16, 16)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(ulong2, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(ulong4, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(ulong8, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(ulong16, 16)
// half
#ifdef cl_khr_fp16
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(half2, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(half4, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(half8, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(half16, 16)
#endif // cl_khr_fp16
// float
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(float2, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(float4, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(float8, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(float16, 16)
// double
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(double2, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(double4, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(double8, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(double16, 16)
#undef __AMDGCN_CLC_SUBGROUP_UP_TO_VEC

#define __AMDGCN_CLC_SUBGROUP_UP_TO_VEC(TYPE, MANGLED_SCALAR_TY, NUM_ELEMS)               \
  _CLC_DEF TYPE                                                                           \
      _Z30__spirv_SubgroupShuffleUpINTELIDv##NUM_ELEMS##_##MANGLED_SCALAR_TY##ET_S1_S1_j( \
          TYPE previous, TYPE current, unsigned int delta) {                              \
    return __spirv_SubgroupShuffleUpINTEL(previous, current, delta);                      \
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
// half
#ifdef cl_khr_fp16
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(half2, Dh, 2)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(half4, Dh, 4)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(half8, Dh, 8)
__AMDGCN_CLC_SUBGROUP_UP_TO_VEC(half16, Dh, 16)
#endif // cl_khr_fp16
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
_CLC_OVERLOAD _CLC_DEF int
__spirv_SubgroupShuffleDownINTEL(int current, int next, unsigned int delta) {
  int self = SELF;
  int size = SUBGROUP_SIZE;

  int index = self + delta;

  int val;
  if (index < size) {
    val = current;
  } else if (index < 2 * size) {
    val = next;
    index = index - size;
  } else {
    // index out of bounds so return arbitrary data
    val = current;
    index = self;
  }

  return __builtin_amdgcn_ds_bpermute(index << 2, val);
}
// int __spirv_SubgroupShuffleDownINTEL<int>(int, int, unsigned int)
_CLC_DEF int
_Z32__spirv_SubgroupShuffleDownINTELIiET_S0_S0_j(int current, int next,
                                                 unsigned int delta) {
  return __spirv_SubgroupShuffleDownINTEL(current, next, delta);
}

// Sub 32-bit types.
#define __AMDGCN_CLC_SUBGROUP_DOWN_TO_I32(TYPE)                                \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleDownINTEL(                \
      TYPE current, TYPE next, unsigned int delta) {                           \
    return __spirv_SubgroupShuffleDownINTEL((int)current, (int)next, delta);   \
  }
__AMDGCN_CLC_SUBGROUP_DOWN_TO_I32(char);
__AMDGCN_CLC_SUBGROUP_DOWN_TO_I32(unsigned char);
__AMDGCN_CLC_SUBGROUP_DOWN_TO_I32(short);
__AMDGCN_CLC_SUBGROUP_DOWN_TO_I32(unsigned short);
// half
#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DEF half __spirv_SubgroupShuffleDownINTEL(
    half current, half next, unsigned int delta) {
  unsigned short tmpC = __clc_as_ushort(current);
  unsigned short tmpN = __clc_as_ushort(next);
  tmpC = __spirv_SubgroupShuffleDownINTEL(tmpC, tmpN, delta);
  return __clc_as_half(tmpC);
}
#endif // cl_khr_fp16
#undef __AMDGCN_CLC_SUBGROUP_DOWN_TO_I32

// _Z32__spirv_SubgroupShuffleDownINTELIaET_S0_S0_j - char
// _Z32__spirv_SubgroupShuffleDownINTELIhET_S0_S0_j - unsigned char
// _Z32__spirv_SubgroupShuffleDownINTELIsET_S0_S0_j - short
// _Z32__spirv_SubgroupShuffleDownINTELItET_S0_S0_j - unsigned short
// _Z32__spirv_SubgroupShuffleDownINTELIDhET_S0_S0_j - half
#define __AMDGCN_CLC_SUBGROUP_DOWN_TO_I32(TYPE, MANGLED_TYPE_NAME)             \
  _CLC_DEF TYPE                                                                \
      _Z32__spirv_SubgroupShuffleDownINTELI##MANGLED_TYPE_NAME##ET_S0_S0_j(    \
          TYPE current, TYPE next, unsigned int delta) {                       \
    return __spirv_SubgroupShuffleDownINTEL(current, next, delta);             \
  }
__AMDGCN_CLC_SUBGROUP_DOWN_TO_I32(char, a);
__AMDGCN_CLC_SUBGROUP_DOWN_TO_I32(unsigned char, h);
__AMDGCN_CLC_SUBGROUP_DOWN_TO_I32(short, s);
__AMDGCN_CLC_SUBGROUP_DOWN_TO_I32(unsigned short, t);
// half
#ifdef cl_khr_fp16
_CLC_DEF half _Z32__spirv_SubgroupShuffleDownINTELIDhET_S0_S0_j(
    half current, half next, unsigned int delta) {
  return __spirv_SubgroupShuffleDownINTEL(current, next, delta);
}
#endif // cl_khr_fp16
#undef __AMDGCN_CLC_SUBGROUP_DOWN_TO_I32

// 32-bit types.
#define __AMDGCN_CLC_SUBGROUP_DOWN_I32(TYPE, CAST_TYPE)                        \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleDownINTEL(                \
      TYPE current, TYPE next, unsigned int delta) {                           \
    return __builtin_astype(                                                   \
        __spirv_SubgroupShuffleDownINTEL(__clc_as_int(current),                \
                                         __clc_as_int(next), delta),           \
        CAST_TYPE);                                                            \
  }
__AMDGCN_CLC_SUBGROUP_DOWN_I32(unsigned int, uint);
__AMDGCN_CLC_SUBGROUP_DOWN_I32(float, float);
#undef __AMDGCN_CLC_SUBGROUP_DOWN_I32

// __spirv_SubgroupShuffleDownINTEL - unsigned int
// __spirv_SubgroupShuffleDownINTEL - float
#define __AMDGCN_CLC_SUBGROUP_DOWN_I32(TYPE, CAST_TYPE, MANGLED_TYPE_NAME)     \
  _CLC_DEF TYPE                                                                \
      _Z32__spirv_SubgroupShuffleDownINTELI##MANGLED_TYPE_NAME##ET_S0_S0_j(    \
          TYPE current, TYPE next, unsigned int delta) {                       \
    return __spirv_SubgroupShuffleDownINTEL(current, next, delta);             \
  }
__AMDGCN_CLC_SUBGROUP_DOWN_I32(unsigned int, uint, j);
__AMDGCN_CLC_SUBGROUP_DOWN_I32(float, float, f);
#undef __AMDGCN_CLC_SUBGROUP_DOWN_I32

// 64-bit types.
#define __AMDGCN_CLC_SUBGROUP_DOWN_I64(TYPE, CAST_TYPE)                        \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleDownINTEL(                \
      TYPE current, TYPE next, unsigned int delta) {                           \
    int2 tmp_current = __clc_as_int2(current);                                 \
    int2 tmp_next = __clc_as_int2(next);                                       \
    int2 ret;                                                                  \
    ret.lo =                                                                   \
        __spirv_SubgroupShuffleDownINTEL(tmp_current.lo, tmp_next.lo, delta);  \
    ret.hi =                                                                   \
        __spirv_SubgroupShuffleDownINTEL(tmp_current.hi, tmp_next.hi, delta);  \
    return __builtin_astype(ret, CAST_TYPE);                                   \
  }
__AMDGCN_CLC_SUBGROUP_DOWN_I64(long, long);
__AMDGCN_CLC_SUBGROUP_DOWN_I64(unsigned long, ulong);
__AMDGCN_CLC_SUBGROUP_DOWN_I64(double, double);
#undef __AMDGCN_CLC_SUBGROUP_DOWN_I64

// double __spirv_SubgroupShuffleDownINTEL<double>(double, unsigned int, int)
#define __AMDGCN_CLC_SUBGROUP_DOWN_I64(TYPE, CAST_TYPE, MANGLED_TYPE_NAME)     \
  _CLC_DEF TYPE                                                                \
      _Z32__spirv_SubgroupShuffleDownINTELI##MANGLED_TYPE_NAME##ET_S0_S0_j(    \
          TYPE current, TYPE next, unsigned int delta) {                       \
    return __spirv_SubgroupShuffleDownINTEL(current, next, delta);             \
  }
__AMDGCN_CLC_SUBGROUP_DOWN_I64(long, long, l);
__AMDGCN_CLC_SUBGROUP_DOWN_I64(unsigned long, ulong, m);
__AMDGCN_CLC_SUBGROUP_DOWN_I64(double, double, d);
#undef __AMDGCN_CLC_SUBGROUP_DOWN_I64

// Vector types.
#define __AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(TYPE, NUM_ELEMS)                     \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleDownINTEL(                \
      TYPE current, TYPE next, unsigned int delta) {                           \
    TYPE res;                                                                  \
    for (int i = 0; i < NUM_ELEMS; ++i) {                                      \
      res[i] = __spirv_SubgroupShuffleDownINTEL(current[i], next[i], delta);   \
    }                                                                          \
    return res;                                                                \
  }
// [u]char
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(char2, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(char4, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(char8, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(char16, 16)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(uchar2, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(uchar4, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(uchar8, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(uchar16, 16)
// [u]short
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(short2, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(short4, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(short8, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(short16, 16)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(ushort2, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(ushort4, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(ushort8, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(ushort16, 16)
// [u]int
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(int2, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(int4, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(int8, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(int16, 16)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(uint2, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(uint4, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(uint8, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(uint16, 16)
// [u]long
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(long2, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(long4, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(long8, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(long16, 16)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(ulong2, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(ulong4, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(ulong8, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(ulong16, 16)
// half
#ifdef cl_khr_fp16
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(half2, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(half4, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(half8, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(half16, 16)
#endif // cl_khr_fp16
// float
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(float2, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(float4, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(float8, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(float16, 16)
// double
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(double2, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(double4, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(double8, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(double16, 16)
#undef __AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC

#define __AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(TYPE, MANGLED_SCALAR_TY, NUM_ELEMS)               \
  _CLC_DEF TYPE                                                                             \
      _Z32__spirv_SubgroupShuffleDownINTELIDv##NUM_ELEMS##_##MANGLED_SCALAR_TY##ET_S1_S1_j( \
          TYPE current, TYPE next, unsigned int delta) {                                    \
    return __spirv_SubgroupShuffleDownINTEL(current, next, delta);                          \
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
// half
#ifdef cl_khr_fp16
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(half2, Dh, 2)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(half4, Dh, 4)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(half8, Dh, 8)
__AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(half16, Dh, 16)
#endif // cl_khr_fp16
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
