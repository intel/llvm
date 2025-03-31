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

// Sub 32-bit types.
#define __AMDGCN_CLC_SUBGROUP_SUB_I32(TYPE)                       \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleINTEL(       \
      TYPE Data, unsigned int InvocationId) {                     \
    return __spirv_SubgroupShuffleINTEL((int)Data, InvocationId); \
  }
__AMDGCN_CLC_SUBGROUP_SUB_I32(char);
__AMDGCN_CLC_SUBGROUP_SUB_I32(unsigned char);
__AMDGCN_CLC_SUBGROUP_SUB_I32(short);
__AMDGCN_CLC_SUBGROUP_SUB_I32(unsigned short);

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
_CLC_OVERLOAD _CLC_DEF half __spirv_SubgroupShuffleINTEL(
    half Data, unsigned int InvocationId) {
  unsigned short tmp = as_ushort(Data);
  tmp = __spirv_SubgroupShuffleINTEL(tmp, InvocationId);
  return as_half(tmp);
}
#endif // cl_khr_fp16

#undef __AMDGCN_CLC_SUBGROUP_SUB_I32

// 32-bit types.
#define __AMDGCN_CLC_SUBGROUP_I32(TYPE, CAST_TYPE)                \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleINTEL(       \
      TYPE Data, unsigned int InvocationId) {                     \
    return __builtin_astype(                                      \
        __spirv_SubgroupShuffleINTEL(as_int(Data), InvocationId), \
        CAST_TYPE);                                               \
  }
__AMDGCN_CLC_SUBGROUP_I32(unsigned int, uint);
__AMDGCN_CLC_SUBGROUP_I32(float, float);
#undef __AMDGCN_CLC_SUBGROUP_I32

// 64-bit types.
#define __AMDGCN_CLC_SUBGROUP_I64(TYPE, CAST_TYPE)                \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleINTEL(       \
      TYPE Data, unsigned int InvocationId) {                     \
    int2 tmp = as_int2(Data);                                     \
    tmp.lo = __spirv_SubgroupShuffleINTEL(tmp.lo, InvocationId);  \
    tmp.hi = __spirv_SubgroupShuffleINTEL(tmp.hi, InvocationId);  \
    return __builtin_astype(tmp, CAST_TYPE);                      \
  }
__AMDGCN_CLC_SUBGROUP_I64(long, long);
__AMDGCN_CLC_SUBGROUP_I64(unsigned long, ulong);
__AMDGCN_CLC_SUBGROUP_I64(double, double);
#undef __AMDGCN_CLC_SUBGROUP_I64

// Vector types.
#define __AMDGCN_CLC_SUBGROUP_TO_VEC(TYPE, NUM_ELEMS)                  \
  _CLC_OVERLOAD _CLC_DEF TYPE                                          \
  __spirv_SubgroupShuffleINTEL(TYPE Data, unsigned int InvocationId) { \
    TYPE res;                                                          \
    for (int i = 0; i < NUM_ELEMS; ++i) {                              \
      res[i] = __spirv_SubgroupShuffleINTEL(Data[i], InvocationId);    \
    }                                                                  \
    return res;                                                        \
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

// Shuffle XOR
_CLC_OVERLOAD _CLC_DEF int
__spirv_SubgroupShuffleXorINTEL(int Data, unsigned int InvocationId) {
  int self = SELF;
  unsigned int index = self ^ InvocationId;
  index =
      index >= ((self + SUBGROUP_SIZE) & ~(SUBGROUP_SIZE - 1)) ? self : index;
  return __builtin_amdgcn_ds_bpermute(index << 2, Data);
}

// Sub 32-bit types.
#define __AMDGCN_CLC_SUBGROUP_XOR_SUB_I32(TYPE)                       \
  _CLC_OVERLOAD _CLC_DEF TYPE                                         \
  __spirv_SubgroupShuffleXor(TYPE Data, unsigned int InvocationId) {  \
    return __spirv_SubgroupShuffleXorINTEL((int)Data, InvocationId);  \
  }
__AMDGCN_CLC_SUBGROUP_XOR_SUB_I32(char);
__AMDGCN_CLC_SUBGROUP_XOR_SUB_I32(unsigned char);
__AMDGCN_CLC_SUBGROUP_XOR_SUB_I32(short);
__AMDGCN_CLC_SUBGROUP_XOR_SUB_I32(unsigned short);
#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DEF half __spirv_SubgroupShuffleXorINTEL(
    half Data, unsigned int InvocationId) {
  unsigned short tmp = as_ushort(Data);
  tmp = (unsigned short)__spirv_SubgroupShuffleXorINTEL(tmp, InvocationId);
  return as_half(tmp);
}
#endif // cl_khr_fp16
#undef __AMDGCN_CLC_SUBGROUP_XOR_SUB_I32

// 32-bit types.
// __spirv_SubgroupShuffleXorINTEL - unsigned int
// __spirv_SubgroupShuffleXorINTEL - float
#define __AMDGCN_CLC_SUBGROUP_XOR_I32(TYPE, CAST_TYPE)                     \
  _CLC_OVERLOAD _CLC_DEF TYPE                                              \
  __spirv_SubgroupShuffleXorINTEL(TYPE Data, unsigned int InvocationId) {  \
    return __builtin_astype(__spirv_SubgroupShuffleXorINTEL(               \
                                as_int(Data), InvocationId),               \
                            CAST_TYPE);                                    \
  }
__AMDGCN_CLC_SUBGROUP_XOR_I32(unsigned int, uint);
__AMDGCN_CLC_SUBGROUP_XOR_I32(float, float);
#undef __AMDGCN_CLC_SUBGROUP_XOR_I32

// 64-bit types.
#define __AMDGCN_CLC_SUBGROUP_XOR_I64(TYPE, CAST_TYPE)                    \
  _CLC_OVERLOAD _CLC_DEF TYPE                                             \
  __spirv_SubgroupShuffleXorINTEL(TYPE Data, unsigned int InvocationId) { \
    int2 tmp = as_int2(Data);                                             \
    tmp.lo = __spirv_SubgroupShuffleXorINTEL(tmp.lo, InvocationId);       \
    tmp.hi = __spirv_SubgroupShuffleXorINTEL(tmp.hi, InvocationId);       \
    return __builtin_astype(tmp, CAST_TYPE);                              \
  }
__AMDGCN_CLC_SUBGROUP_XOR_I64(long, long);
__AMDGCN_CLC_SUBGROUP_XOR_I64(unsigned long, ulong);
__AMDGCN_CLC_SUBGROUP_XOR_I64(double, double);
#undef __AMDGCN_CLC_SUBGROUP_XOR_I64

// Vector types.
#define __AMDGCN_CLC_SUBGROUP_XOR_TO_VEC(TYPE, NUM_ELEMS)                  \
  _CLC_OVERLOAD _CLC_DEF TYPE                                              \
  __spirv_SubgroupShuffleXorINTEL(TYPE Data, unsigned int InvocationId) {  \
    TYPE res;                                                              \
    for (int i = 0; i < NUM_ELEMS; ++i) {                                  \
      res[i] = __spirv_SubgroupShuffleXorINTEL(Data[i], InvocationId);     \
    }                                                                      \
    return res;                                                            \
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
  unsigned short tmpP = as_ushort(previous);
  unsigned short tmpC = as_ushort(current);
  tmpC = __spirv_SubgroupShuffleUpINTEL(tmpP, tmpC, delta);
  return as_half(tmpC);
}
#endif // cl_khr_fp16
#undef __AMDGCN_CLC_SUBGROUP_UP_SUB_I32

// 32-bit types.
#define __AMDGCN_CLC_SUBGROUP_UP_I32(TYPE, CAST_TYPE)                      \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleUpINTEL(              \
      TYPE previous, TYPE current, unsigned int delta) {                   \
    return __builtin_astype(__spirv_SubgroupShuffleUpINTEL(                \
                                as_int(previous), as_int(current), delta), \
                            CAST_TYPE);                                    \
  }
__AMDGCN_CLC_SUBGROUP_UP_I32(unsigned int, uint);
__AMDGCN_CLC_SUBGROUP_UP_I32(float, float);
#undef __AMDGCN_CLC_SUBGROUP_UP_I32

// 64-bit types.
#define __AMDGCN_CLC_SUBGROUP_UP_I64(TYPE, CAST_TYPE)         \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleUpINTEL( \
          TYPE previous, TYPE current, unsigned int delta) {  \
    int2 tmp_previous = as_int2(previous);                    \
    int2 tmp_current = as_int2(current);                      \
    int2 ret;                                                 \
    ret.lo = __spirv_SubgroupShuffleUpINTEL(                  \
        tmp_previous.lo, tmp_current.lo, delta);              \
    ret.hi = __spirv_SubgroupShuffleUpINTEL(                  \
        tmp_previous.hi, tmp_current.hi, delta);              \
    return __builtin_astype(ret, CAST_TYPE);                  \
  }
__AMDGCN_CLC_SUBGROUP_UP_I64(long, long);
__AMDGCN_CLC_SUBGROUP_UP_I64(unsigned long, ulong);
__AMDGCN_CLC_SUBGROUP_UP_I64(double, double);
#undef __AMDGCN_CLC_SUBGROUP_UP_I64

// Vector types.
#define __AMDGCN_CLC_SUBGROUP_UP_TO_VEC(TYPE, NUM_ELEMS)                       \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleUpINTEL(                  \
          TYPE previous, TYPE current, unsigned int delta) {                   \
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

// Sub 32-bit types.
#define __AMDGCN_CLC_SUBGROUP_DOWN_TO_I32(TYPE)                              \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleDownINTEL(              \
          TYPE current, TYPE next, unsigned int delta) {                     \
    return __spirv_SubgroupShuffleDownINTEL((int)current, (int)next, delta); \
  }
__AMDGCN_CLC_SUBGROUP_DOWN_TO_I32(char);
__AMDGCN_CLC_SUBGROUP_DOWN_TO_I32(unsigned char);
__AMDGCN_CLC_SUBGROUP_DOWN_TO_I32(short);
__AMDGCN_CLC_SUBGROUP_DOWN_TO_I32(unsigned short);
// half
#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DEF half __spirv_SubgroupShuffleDownINTEL(
    half current, half next, unsigned int delta) {
  unsigned short tmpC = as_ushort(current);
  unsigned short tmpN = as_ushort(next);
  tmpC = __spirv_SubgroupShuffleDownINTEL(tmpC, tmpN, delta);
  return as_half(tmpC);
}
#endif // cl_khr_fp16
#undef __AMDGCN_CLC_SUBGROUP_DOWN_TO_I32

// 32-bit types.
// __spirv_SubgroupShuffleDownINTEL - unsigned int
// __spirv_SubgroupShuffleDownINTEL - float
#define __AMDGCN_CLC_SUBGROUP_DOWN_I32(TYPE, CAST_TYPE)                \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleDownINTEL(        \
          TYPE current, TYPE next, unsigned int delta) {               \
    return __builtin_astype(__spirv_SubgroupShuffleDownINTEL(          \
                                as_int(current), as_int(next), delta), \
                            CAST_TYPE);                                \
  }
__AMDGCN_CLC_SUBGROUP_DOWN_I32(unsigned int, uint);
__AMDGCN_CLC_SUBGROUP_DOWN_I32(float, float);
#undef __AMDGCN_CLC_SUBGROUP_DOWN_I32

// 64-bit types.
#define __AMDGCN_CLC_SUBGROUP_DOWN_I64(TYPE, CAST_TYPE)                        \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleDownINTEL(                \
          TYPE current, TYPE next, unsigned int delta) {                       \
    int2 tmp_current = as_int2(current);                                       \
    int2 tmp_next = as_int2(next);                                             \
    int2 ret;                                                                  \
    ret.lo = __spirv_SubgroupShuffleDownINTEL(                                 \
        tmp_current.lo, tmp_next.lo, delta);                                   \
    ret.hi = __spirv_SubgroupShuffleDownINTEL(                                 \
        tmp_current.hi, tmp_next.hi, delta);                                   \
    return __builtin_astype(ret, CAST_TYPE);                                   \
  }
__AMDGCN_CLC_SUBGROUP_DOWN_I64(long, long);
__AMDGCN_CLC_SUBGROUP_DOWN_I64(unsigned long, ulong);
__AMDGCN_CLC_SUBGROUP_DOWN_I64(double, double);
#undef __AMDGCN_CLC_SUBGROUP_DOWN_I64

// Vector types.
#define __AMDGCN_CLC_SUBGROUP_DOWN_TO_VEC(TYPE, NUM_ELEMS)                   \
  _CLC_OVERLOAD _CLC_DEF TYPE __spirv_SubgroupShuffleDownINTEL(              \
          TYPE current, TYPE next, unsigned int delta) {                     \
    TYPE res;                                                                \
    for (int i = 0; i < NUM_ELEMS; ++i) {                                    \
      res[i] = __spirv_SubgroupShuffleDownINTEL(current[i], next[i], delta); \
    }                                                                        \
    return res;                                                              \
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
