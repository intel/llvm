//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "membermask.h"

#include <libspirv/spirv.h>
#include <libspirv/spirv_types.h>

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

int __clc_nvvm_reflect_arch();

// CLC helpers
__local bool *
__clc__get_group_scratch_bool() __asm("__clc__get_group_scratch_bool");
__local char *
__clc__get_group_scratch_char() __asm("__clc__get_group_scratch_char");
__local uchar *
__clc__get_group_scratch_uchar() __asm("__clc__get_group_scratch_char");
__local short *
__clc__get_group_scratch_short() __asm("__clc__get_group_scratch_short");
__local ushort *
__clc__get_group_scratch_ushort() __asm("__clc__get_group_scratch_short");
__local int *
__clc__get_group_scratch_int() __asm("__clc__get_group_scratch_int");
__local uint *
__clc__get_group_scratch_uint() __asm("__clc__get_group_scratch_int");
__local long *
__clc__get_group_scratch_long() __asm("__clc__get_group_scratch_long");
__local ulong *
__clc__get_group_scratch_ulong() __asm("__clc__get_group_scratch_long");
__local half *
__clc__get_group_scratch_half() __asm("__clc__get_group_scratch_half");
__local float *
__clc__get_group_scratch_float() __asm("__clc__get_group_scratch_float");
__local double *
__clc__get_group_scratch_double() __asm("__clc__get_group_scratch_double");
__local complex_half *__clc__get_group_scratch_complex_half() __asm(
    "__clc__get_group_scratch_complex_half");
__local complex_float *__clc__get_group_scratch_complex_float() __asm(
    "__clc__get_group_scratch_complex_float");
__local complex_double *__clc__get_group_scratch_complex_double() __asm(
    "__clc__get_group_scratch_complex_double");

_CLC_DEF uint inline __clc__membermask() {
  // use a full mask as sync operations are required to be convergent and
  // exited threads can safely be in the mask
  return 0xFFFFFFFF;
}

#define __CLC_SUBGROUP_SHUFFLE_I32(TYPE)                                       \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT TYPE __clc__SubgroupShuffle(          \
      TYPE x, uint idx) {                                                      \
    return __nvvm_shfl_sync_idx_i32(__clc__membermask(), x, idx, 0x1f);        \
  }
__CLC_SUBGROUP_SHUFFLE_I32(char);
__CLC_SUBGROUP_SHUFFLE_I32(uchar);
__CLC_SUBGROUP_SHUFFLE_I32(short);
__CLC_SUBGROUP_SHUFFLE_I32(ushort);
__CLC_SUBGROUP_SHUFFLE_I32(int);
__CLC_SUBGROUP_SHUFFLE_I32(uint);
#undef __CLC_SUBGROUP_SHUFFLE_I32

_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT ulong __clc__SubgroupShuffle(ulong x,
                                                                    uint idx) {
  uint2 y = as_uint2(x);
  y.lo = __nvvm_shfl_sync_idx_i32(__clc__membermask(), y.lo, idx, 0x1f);
  y.hi = __nvvm_shfl_sync_idx_i32(__clc__membermask(), y.hi, idx, 0x1f);
  return as_ulong(y);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT long __clc__SubgroupShuffle(long x,
                                                                   uint idx) {
  return as_long(__clc__SubgroupShuffle(as_ulong(x), idx));
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT half __clc__SubgroupShuffle(half x,
                                                                   uint idx) {
  return as_half(__clc__SubgroupShuffle(as_short(x), idx));
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT float __clc__SubgroupShuffle(float x,
                                                                    uint idx) {
  return __nvvm_shfl_sync_idx_f32(__clc__membermask(), x, idx, 0x1f);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT double __clc__SubgroupShuffle(double x,
                                                                     uint idx) {
  return as_double(__clc__SubgroupShuffle(as_ulong(x), idx));
}

typedef union {
  complex_half h;
  int i;
} complex_half_converter;

typedef union {
  complex_float f;
  int2 i;
} complex_float_converter;

typedef union {
  complex_double d;
  int4 i;
} complex_double_converter;

_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT complex_half
__clc__SubgroupShuffle(complex_half x, uint idx) {
  complex_half_converter conv = {x};
  conv.i = __clc__SubgroupShuffle(conv.i, idx);
  return conv.h;
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT complex_float
__clc__SubgroupShuffle(complex_float x, uint idx) {
  complex_float_converter conv = {x};
  conv.i.x = __clc__SubgroupShuffle(conv.i.x, idx);
  conv.i.y = __clc__SubgroupShuffle(conv.i.y, idx);
  return conv.f;
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT complex_double
__clc__SubgroupShuffle(complex_double x, uint idx) {
  complex_double_converter conv = {x};
  conv.i.x = __clc__SubgroupShuffle(conv.i.x, idx);
  conv.i.y = __clc__SubgroupShuffle(conv.i.y, idx);
  conv.i.z = __clc__SubgroupShuffle(conv.i.z, idx);
  conv.i.w = __clc__SubgroupShuffle(conv.i.w, idx);
  return conv.d;
}

#define __CLC_SUBGROUP_SHUFFLEUP_I32(TYPE)                                     \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT TYPE __clc__SubgroupShuffleUp(        \
      TYPE x, uint delta) {                                                    \
    return __nvvm_shfl_sync_up_i32(__clc__membermask(), x, delta, 0);          \
  }
__CLC_SUBGROUP_SHUFFLEUP_I32(char);
__CLC_SUBGROUP_SHUFFLEUP_I32(uchar);
__CLC_SUBGROUP_SHUFFLEUP_I32(short);
__CLC_SUBGROUP_SHUFFLEUP_I32(ushort);
__CLC_SUBGROUP_SHUFFLEUP_I32(int);
__CLC_SUBGROUP_SHUFFLEUP_I32(uint);
#undef __CLC_SUBGROUP_SHUFFLEUP_I32

_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT ulong
__clc__SubgroupShuffleUp(ulong x, uint delta) {
  uint2 y = as_uint2(x);
  y.lo = __nvvm_shfl_sync_up_i32(__clc__membermask(), y.lo, delta, 0);
  y.hi = __nvvm_shfl_sync_up_i32(__clc__membermask(), y.hi, delta, 0);
  return as_ulong(y);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT long
__clc__SubgroupShuffleUp(long x, uint delta) {
  return as_long(__clc__SubgroupShuffleUp(as_ulong(x), delta));
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT half
__clc__SubgroupShuffleUp(half x, uint delta) {
  return as_half(__clc__SubgroupShuffleUp(as_short(x), delta));
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT float
__clc__SubgroupShuffleUp(float x, uint delta) {
  return __nvvm_shfl_sync_up_f32(__clc__membermask(), x, delta, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT double
__clc__SubgroupShuffleUp(double x, uint delta) {
  return as_double(__clc__SubgroupShuffleUp(as_ulong(x), delta));
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT complex_half
__clc__SubgroupShuffleUp(complex_half x, uint delta) {
  complex_half_converter conv = {x};
  conv.i = __clc__SubgroupShuffleUp(conv.i, delta);
  return conv.h;
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT complex_float
__clc__SubgroupShuffleUp(complex_float x, uint delta) {
  complex_float_converter conv = {x};
  conv.i.x = __clc__SubgroupShuffleUp(conv.i.x, delta);
  conv.i.y = __clc__SubgroupShuffleUp(conv.i.y, delta);
  return conv.f;
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT complex_double
__clc__SubgroupShuffleUp(complex_double x, uint delta) {
  complex_double_converter conv = {x};
  conv.i.x = __clc__SubgroupShuffleUp(conv.i.x, delta);
  conv.i.y = __clc__SubgroupShuffleUp(conv.i.y, delta);
  conv.i.z = __clc__SubgroupShuffleUp(conv.i.z, delta);
  conv.i.w = __clc__SubgroupShuffleUp(conv.i.w, delta);
  return conv.d;
}

// TODO: Implement InclusiveScan/ExclusiveScan
//       Currently only Reduce is required (for GroupAny and GroupAll)
_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT bool
__clc__SubgroupBitwiseOr(uint op, bool predicate, bool *carry) {
  bool result = __nvvm_vote_any_sync(__clc__membermask(), predicate);
  *carry = result;
  return result;
}
_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT bool
__clc__SubgroupBitwiseAny(uint op, bool predicate, bool *carry) {
  bool result = __nvvm_vote_all_sync(__clc__membermask(), predicate);
  *carry = result;
  return result;
}

#define __CLC_APPEND(NAME, SUFFIX) NAME##SUFFIX

#define __CLC_ADD(x, y) (x + y)
#define __CLC_MIN(x, y) ((x < y) ? (x) : (y))
#define __CLC_MAX(x, y) ((x > y) ? (x) : (y))
#define __CLC_OR(x, y) (x | y)
#define __CLC_XOR(x, y) (x ^ y)
#define __CLC_AND(x, y) (x & y)
#define __CLC_MUL(x, y) (x * y)
#define __CLC_LOGICAL_OR(x, y) (x || y)
#define __CLC_LOGICAL_AND(x, y) (x && y)

#define __DEFINE_CLC_COMPLEX_MUL(TYPE)                                         \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT complex_##TYPE __clc_complex_mul(     \
      complex_##TYPE z, complex_##TYPE w) {                                    \
    TYPE a = z.real;                                                           \
    TYPE b = z.imag;                                                           \
    TYPE c = w.real;                                                           \
    TYPE d = w.imag;                                                           \
    TYPE ac = a * c;                                                           \
    TYPE bd = b * d;                                                           \
    TYPE ad = a * d;                                                           \
    TYPE bc = b * c;                                                           \
    TYPE x = ac - bd;                                                          \
    TYPE y = ad + bc;                                                          \
    if (__spirv_IsNan(x) && __spirv_IsNan(y)) {                                \
      bool __recalc = false;                                                   \
      if (__spirv_IsInf(a) || __spirv_IsInf(b)) {                              \
        a = __spirv_ocl_copysign(__spirv_IsInf(a) ? (TYPE)1 : (TYPE)0, a);     \
        b = __spirv_ocl_copysign(__spirv_IsInf(b) ? (TYPE)1 : (TYPE)0, b);     \
        if (__spirv_IsNan(c))                                                  \
          c = __spirv_ocl_copysign((TYPE)0, c);                                \
        if (__spirv_IsNan(d))                                                  \
          d = __spirv_ocl_copysign((TYPE)0, d);                                \
        __recalc = true;                                                       \
      }                                                                        \
      if (__spirv_IsInf(c) || __spirv_IsInf(d)) {                              \
        c = __spirv_ocl_copysign(__spirv_IsInf(c) ? (TYPE)1 : (TYPE)0, c);     \
        d = __spirv_ocl_copysign(__spirv_IsInf(d) ? (TYPE)1 : (TYPE)0, d);     \
        if (__spirv_IsNan(a))                                                  \
          a = __spirv_ocl_copysign((TYPE)0, a);                                \
        if (__spirv_IsNan(b))                                                  \
          b = __spirv_ocl_copysign((TYPE)0, b);                                \
        __recalc = true;                                                       \
      }                                                                        \
      if (!__recalc && (__spirv_IsInf(ac) || __spirv_IsInf(bd) ||              \
                        __spirv_IsInf(ad) || __spirv_IsInf(bc))) {             \
        if (__spirv_IsNan(a))                                                  \
          a = __spirv_ocl_copysign((TYPE)0, a);                                \
        if (__spirv_IsNan(b))                                                  \
          b = __spirv_ocl_copysign((TYPE)0, b);                                \
        if (__spirv_IsNan(c))                                                  \
          c = __spirv_ocl_copysign((TYPE)0, c);                                \
        if (__spirv_IsNan(d))                                                  \
          d = __spirv_ocl_copysign((TYPE)0, d);                                \
        __recalc = true;                                                       \
      }                                                                        \
      if (__recalc) {                                                          \
        x = (TYPE)INFINITY * (a * c - b * d);                                  \
        y = (TYPE)INFINITY * (a * d + b * c);                                  \
      }                                                                        \
    }                                                                          \
    return (complex_##TYPE){x, y};                                             \
  }

__DEFINE_CLC_COMPLEX_MUL(half)
__DEFINE_CLC_COMPLEX_MUL(float)
__DEFINE_CLC_COMPLEX_MUL(double)
#undef __DEFINE_CLC_COMPLEX_MUL

// TODO remove these definitions after we have proper implementation of
// std::complex multiplication in SYCL
complex_float __mulsc3(float a, float b, float c, float d) {
  return __clc_complex_mul((complex_float){a, b}, (complex_float){c, d});
}
complex_double __muldc3(double a, double b, double c, double d) {
  return __clc_complex_mul((complex_double){a, b}, (complex_double){c, d});
}

#define __CLC_COMPLEX_MUL(x, y) __clc_complex_mul(x, y)

#define __CLC_SUBGROUP_COLLECTIVE_BODY(OP, TYPE, IDENTITY)                     \
  uint sg_lid = __spirv_SubgroupLocalInvocationId();                           \
  /* Can't use XOR/butterfly shuffles; some lanes may be inactive */           \
  for (int o = 1; o < __spirv_SubgroupMaxSize(); o *= 2) {                     \
    TYPE contribution = __clc__SubgroupShuffleUp(x, o);                        \
    bool inactive = (sg_lid < o);                                              \
    contribution = (inactive) ? IDENTITY : contribution;                       \
    x = OP(x, contribution);                                                   \
  }                                                                            \
  /* For Reduce, broadcast result from highest active lane */                  \
  TYPE result;                                                                 \
  if (op == Reduce) {                                                          \
    result = __clc__SubgroupShuffle(x, __spirv_SubgroupSize() - 1);            \
    *carry = result;                                                           \
  } /* For InclusiveScan, use results as computed */                           \
  else if (op == InclusiveScan) {                                              \
    result = x;                                                                \
    *carry = result;                                                           \
  } /* For ExclusiveScan, shift and prepend identity */                        \
  else if (op == ExclusiveScan) {                                              \
    *carry = x;                                                                \
    result = __clc__SubgroupShuffleUp(x, 1);                                   \
    if (sg_lid == 0) {                                                         \
      result = IDENTITY;                                                       \
    }                                                                          \
  }                                                                            \
  return result;

#define __CLC_SUBGROUP_COLLECTIVE(NAME, OP, TYPE, IDENTITY)                    \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT TYPE __CLC_APPEND(                    \
      __clc__Subgroup, NAME)(uint op, TYPE x, TYPE * carry) {                  \
    __CLC_SUBGROUP_COLLECTIVE_BODY(OP, TYPE, IDENTITY)                         \
  }

#define __CLC_SUBGROUP_COLLECTIVE_REDUX(NAME, OP, REDUX_OP, TYPE, IDENTITY)    \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT TYPE __CLC_APPEND(                    \
      __clc__Subgroup, NAME)(uint op, TYPE x, TYPE * carry) {                  \
    /* Fast path for warp reductions for sm_80+ */                             \
    if (__clc_nvvm_reflect_arch() >= 800 && op == Reduce) {                    \
      TYPE result = __nvvm_redux_sync_##REDUX_OP(x, __clc__membermask());      \
      *carry = result;                                                         \
      return result;                                                           \
    }                                                                          \
    __CLC_SUBGROUP_COLLECTIVE_BODY(OP, TYPE, IDENTITY)                         \
  }

__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, char, 0)
__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, uchar, 0)
__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, short, 0)
__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, ushort, 0)
__CLC_SUBGROUP_COLLECTIVE_REDUX(IAdd, __CLC_ADD, add, int, 0)
__CLC_SUBGROUP_COLLECTIVE_REDUX(IAdd, __CLC_ADD, add, uint, 0)
__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, long, 0)
__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, ulong, 0)
__CLC_SUBGROUP_COLLECTIVE(FAdd, __CLC_ADD, half, 0)
__CLC_SUBGROUP_COLLECTIVE(FAdd, __CLC_ADD, float, 0)
__CLC_SUBGROUP_COLLECTIVE(FAdd, __CLC_ADD, double, 0)

__CLC_SUBGROUP_COLLECTIVE(IMulKHR, __CLC_MUL, char, 1)
__CLC_SUBGROUP_COLLECTIVE(IMulKHR, __CLC_MUL, uchar, 1)
__CLC_SUBGROUP_COLLECTIVE(IMulKHR, __CLC_MUL, short, 1)
__CLC_SUBGROUP_COLLECTIVE(IMulKHR, __CLC_MUL, ushort, 1)
__CLC_SUBGROUP_COLLECTIVE(IMulKHR, __CLC_MUL, int, 1)
__CLC_SUBGROUP_COLLECTIVE(IMulKHR, __CLC_MUL, uint, 1)
__CLC_SUBGROUP_COLLECTIVE(IMulKHR, __CLC_MUL, long, 1)
__CLC_SUBGROUP_COLLECTIVE(IMulKHR, __CLC_MUL, ulong, 1)
__CLC_SUBGROUP_COLLECTIVE(FMulKHR, __CLC_MUL, half, 1)
__CLC_SUBGROUP_COLLECTIVE(FMulKHR, __CLC_MUL, float, 1)
__CLC_SUBGROUP_COLLECTIVE(FMulKHR, __CLC_MUL, double, 1)

__CLC_SUBGROUP_COLLECTIVE(CMulINTEL, __CLC_COMPLEX_MUL, complex_half,
                          ((complex_half){1, 0}))
__CLC_SUBGROUP_COLLECTIVE(CMulINTEL, __CLC_COMPLEX_MUL, complex_float,
                          ((complex_float){1, 0}))
__CLC_SUBGROUP_COLLECTIVE(CMulINTEL, __CLC_COMPLEX_MUL, complex_double,
                          ((complex_double){1, 0}))

__CLC_SUBGROUP_COLLECTIVE(SMin, __CLC_MIN, char, CHAR_MAX)
__CLC_SUBGROUP_COLLECTIVE(UMin, __CLC_MIN, uchar, UCHAR_MAX)
__CLC_SUBGROUP_COLLECTIVE(SMin, __CLC_MIN, short, SHRT_MAX)
__CLC_SUBGROUP_COLLECTIVE(UMin, __CLC_MIN, ushort, USHRT_MAX)
__CLC_SUBGROUP_COLLECTIVE_REDUX(SMin, __CLC_MIN, min, int, INT_MAX)
__CLC_SUBGROUP_COLLECTIVE_REDUX(UMin, __CLC_MIN, umin, uint, UINT_MAX)
__CLC_SUBGROUP_COLLECTIVE(SMin, __CLC_MIN, long, LONG_MAX)
__CLC_SUBGROUP_COLLECTIVE(UMin, __CLC_MIN, ulong, ULONG_MAX)
__CLC_SUBGROUP_COLLECTIVE(FMin, __CLC_MIN, half, INFINITY)
__CLC_SUBGROUP_COLLECTIVE(FMin, __CLC_MIN, float, INFINITY)
__CLC_SUBGROUP_COLLECTIVE(FMin, __CLC_MIN, double, INFINITY)

__CLC_SUBGROUP_COLLECTIVE(SMax, __CLC_MAX, char, CHAR_MIN)
__CLC_SUBGROUP_COLLECTIVE(UMax, __CLC_MAX, uchar, 0)
__CLC_SUBGROUP_COLLECTIVE(SMax, __CLC_MAX, short, SHRT_MIN)
__CLC_SUBGROUP_COLLECTIVE(UMax, __CLC_MAX, ushort, 0)
__CLC_SUBGROUP_COLLECTIVE_REDUX(SMax, __CLC_MAX, max, int, INT_MIN)
__CLC_SUBGROUP_COLLECTIVE_REDUX(UMax, __CLC_MAX, umax, uint, 0)
__CLC_SUBGROUP_COLLECTIVE(SMax, __CLC_MAX, long, LONG_MIN)
__CLC_SUBGROUP_COLLECTIVE(UMax, __CLC_MAX, ulong, 0)
__CLC_SUBGROUP_COLLECTIVE(FMax, __CLC_MAX, half, -INFINITY)
__CLC_SUBGROUP_COLLECTIVE(FMax, __CLC_MAX, float, -INFINITY)
__CLC_SUBGROUP_COLLECTIVE(FMax, __CLC_MAX, double, -INFINITY)

__CLC_SUBGROUP_COLLECTIVE_REDUX(BitwiseAndKHR, __CLC_AND, and, uchar, ~0)
__CLC_SUBGROUP_COLLECTIVE_REDUX(BitwiseOrKHR, __CLC_OR, or, uchar, 0)
__CLC_SUBGROUP_COLLECTIVE_REDUX(BitwiseXorKHR, __CLC_XOR, xor, uchar, 0)
__CLC_SUBGROUP_COLLECTIVE_REDUX(BitwiseAndKHR, __CLC_AND, and, char, ~0)
__CLC_SUBGROUP_COLLECTIVE_REDUX(BitwiseOrKHR, __CLC_OR, or, char, 0)
__CLC_SUBGROUP_COLLECTIVE_REDUX(BitwiseXorKHR, __CLC_XOR, xor, char, 0)

__CLC_SUBGROUP_COLLECTIVE_REDUX(BitwiseAndKHR, __CLC_AND, and, ushort, ~0)
__CLC_SUBGROUP_COLLECTIVE_REDUX(BitwiseOrKHR, __CLC_OR, or, ushort, 0)
__CLC_SUBGROUP_COLLECTIVE_REDUX(BitwiseXorKHR, __CLC_XOR, xor, ushort, 0)
__CLC_SUBGROUP_COLLECTIVE_REDUX(BitwiseAndKHR, __CLC_AND, and, short, ~0)
__CLC_SUBGROUP_COLLECTIVE_REDUX(BitwiseOrKHR, __CLC_OR, or, short, 0)
__CLC_SUBGROUP_COLLECTIVE_REDUX(BitwiseXorKHR, __CLC_XOR, xor, short, 0)

__CLC_SUBGROUP_COLLECTIVE_REDUX(BitwiseAndKHR, __CLC_AND, and, uint, ~0)
__CLC_SUBGROUP_COLLECTIVE_REDUX(BitwiseOrKHR, __CLC_OR, or, uint, 0)
__CLC_SUBGROUP_COLLECTIVE_REDUX(BitwiseXorKHR, __CLC_XOR, xor, uint, 0)
__CLC_SUBGROUP_COLLECTIVE_REDUX(BitwiseAndKHR, __CLC_AND, and, int, ~0)
__CLC_SUBGROUP_COLLECTIVE_REDUX(BitwiseOrKHR, __CLC_OR, or, int, 0)
__CLC_SUBGROUP_COLLECTIVE_REDUX(BitwiseXorKHR, __CLC_XOR, xor, int, 0)

__CLC_SUBGROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, ulong, ~0l)
__CLC_SUBGROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, ulong, 0l)
__CLC_SUBGROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, ulong, 0l)
__CLC_SUBGROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, long, ~0l)
__CLC_SUBGROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, long, 0l)
__CLC_SUBGROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, long, 0l)

__CLC_SUBGROUP_COLLECTIVE(LogicalOrKHR, __CLC_LOGICAL_OR, bool, false)
__CLC_SUBGROUP_COLLECTIVE(LogicalAndKHR, __CLC_LOGICAL_AND, bool, true)

#undef __CLC_SUBGROUP_COLLECTIVE_BODY
#undef __CLC_SUBGROUP_COLLECTIVE
#undef __CLC_SUBGROUP_COLLECTIVE_REDUX

#define __CLC_GROUP_COLLECTIVE_INNER(CLC_NAME, OP, TYPE, IDENTITY)             \
  TYPE carry = IDENTITY;                                                       \
  /* Perform GroupOperation within sub-group */                                \
  TYPE sg_x = __CLC_APPEND(__clc__Subgroup, CLC_NAME)(op, x, &carry);          \
  if (scope == Subgroup) {                                                     \
    return sg_x;                                                               \
  }                                                                            \
  __local TYPE *scratch = __CLC_APPEND(__clc__get_group_scratch_, TYPE)();     \
  uint sg_id = __spirv_SubgroupId();                                           \
  uint num_sg = __spirv_NumSubgroups();                                        \
  uint sg_lid = __spirv_SubgroupLocalInvocationId();                           \
  uint sg_size = __spirv_SubgroupSize();                                       \
  /* Share carry values across sub-groups */                                   \
  if (sg_lid == sg_size - 1) {                                                 \
    scratch[sg_id] = carry;                                                    \
  }                                                                            \
  __spirv_ControlBarrier(Workgroup, 0, 0);                                     \
  /* Perform InclusiveScan over sub-group results */                           \
  TYPE sg_prefix;                                                              \
  TYPE sg_aggregate = scratch[0];                                              \
  for (int s = 1; s < num_sg; ++s) {                                           \
    if (sg_id == s) {                                                          \
      sg_prefix = sg_aggregate;                                                \
    }                                                                          \
    TYPE addend = scratch[s];                                                  \
    sg_aggregate = OP(sg_aggregate, addend);                                   \
  }                                                                            \
  /* For Reduce, broadcast result from final sub-group */                      \
  /* For Scan, combine results from previous sub-groups */                     \
  TYPE result;                                                                 \
  if (op == Reduce) {                                                          \
    result = sg_aggregate;                                                     \
  } else if (op == InclusiveScan || op == ExclusiveScan) {                     \
    if (sg_id == 0) {                                                          \
      result = sg_x;                                                           \
    } else {                                                                   \
      result = OP(sg_x, sg_prefix);                                            \
    }                                                                          \
  }                                                                            \
  __spirv_ControlBarrier(Workgroup, 0, 0);                                     \
  return result;

#define __CLC_GROUP_COLLECTIVE_OUTER(SPIRV_NAME, CLC_NAME, OP, TYPE, IDENTITY) \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT TYPE __CLC_APPEND(                    \
      __spirv_Group, SPIRV_NAME)(uint scope, uint op, TYPE x) {                \
    __CLC_GROUP_COLLECTIVE_INNER(CLC_NAME, OP, TYPE, IDENTITY)                 \
  }

#define __CLC_GROUP_COLLECTIVE_4(NAME, OP, TYPE, IDENTITY)                     \
  __CLC_GROUP_COLLECTIVE_OUTER(NAME, NAME, OP, TYPE, IDENTITY)
#define __CLC_GROUP_COLLECTIVE_5(SPIRV_NAME, CLC_NAME, OP, TYPE, IDENTITY)     \
  __CLC_GROUP_COLLECTIVE_OUTER(SPIRV_NAME, CLC_NAME, OP, TYPE, IDENTITY)

#define DISPATCH_TO_CLC_GROUP_COLLECTIVE_MACRO(_1, _2, _3, _4, _5, NAME, ...)  \
  NAME
#define __CLC_GROUP_COLLECTIVE(...)                                            \
  DISPATCH_TO_CLC_GROUP_COLLECTIVE_MACRO(                                      \
      __VA_ARGS__, __CLC_GROUP_COLLECTIVE_5, __CLC_GROUP_COLLECTIVE_4)         \
  (__VA_ARGS__)

#define __CLC_GROUP_COLLECTIVE_MANUAL_MANGLE(SPIRV_NAME_MANGLED, CLC_NAME, OP, \
                                             TYPE, IDENTITY)                   \
  _CLC_DEF _CLC_CONVERGENT TYPE SPIRV_NAME_MANGLED(uint scope, uint op,        \
                                                   TYPE x) {                   \
    __CLC_GROUP_COLLECTIVE_INNER(CLC_NAME, OP, TYPE, IDENTITY)                 \
  }

__CLC_GROUP_COLLECTIVE(BitwiseOr, __CLC_OR, bool, false);
__CLC_GROUP_COLLECTIVE(BitwiseAny, __CLC_AND, bool, true);
_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT bool __spirv_GroupAny(uint scope,
                                                             bool predicate) {
  return __spirv_GroupBitwiseOr(scope, Reduce, predicate);
}
_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT bool __spirv_GroupAll(uint scope,
                                                             bool predicate) {
  return __spirv_GroupBitwiseAny(scope, Reduce, predicate);
}

__CLC_GROUP_COLLECTIVE(IAdd, __CLC_ADD, char, 0)
__CLC_GROUP_COLLECTIVE(IAdd, __CLC_ADD, uchar, 0)
__CLC_GROUP_COLLECTIVE(IAdd, __CLC_ADD, short, 0)
__CLC_GROUP_COLLECTIVE(IAdd, __CLC_ADD, ushort, 0)
__CLC_GROUP_COLLECTIVE(IAdd, __CLC_ADD, int, 0)
__CLC_GROUP_COLLECTIVE(IAdd, __CLC_ADD, uint, 0)
__CLC_GROUP_COLLECTIVE(IAdd, __CLC_ADD, long, 0)
__CLC_GROUP_COLLECTIVE(IAdd, __CLC_ADD, ulong, 0)
__CLC_GROUP_COLLECTIVE(FAdd, __CLC_ADD, half, 0)
__CLC_GROUP_COLLECTIVE(FAdd, __CLC_ADD, float, 0)
__CLC_GROUP_COLLECTIVE(FAdd, __CLC_ADD, double, 0)

__CLC_GROUP_COLLECTIVE(IMulKHR, __CLC_MUL, char, 1)
__CLC_GROUP_COLLECTIVE(IMulKHR, __CLC_MUL, uchar, 1)
__CLC_GROUP_COLLECTIVE(IMulKHR, __CLC_MUL, short, 1)
__CLC_GROUP_COLLECTIVE(IMulKHR, __CLC_MUL, ushort, 1)
__CLC_GROUP_COLLECTIVE(IMulKHR, __CLC_MUL, int, 1)
__CLC_GROUP_COLLECTIVE(IMulKHR, __CLC_MUL, uint, 1)
__CLC_GROUP_COLLECTIVE(IMulKHR, __CLC_MUL, long, 1)
__CLC_GROUP_COLLECTIVE(IMulKHR, __CLC_MUL, ulong, 1)
__CLC_GROUP_COLLECTIVE(FMulKHR, __CLC_MUL, half, 1)
__CLC_GROUP_COLLECTIVE(FMulKHR, __CLC_MUL, float, 1)
__CLC_GROUP_COLLECTIVE(FMulKHR, __CLC_MUL, double, 1)

__CLC_GROUP_COLLECTIVE_MANUAL_MANGLE(
    _Z22__spirv_GroupCMulINTELjjN5__spv12complex_halfE, CMulINTEL,
    __CLC_COMPLEX_MUL, complex_half, ((complex_half){1, 0}))
__CLC_GROUP_COLLECTIVE_MANUAL_MANGLE(
    _Z22__spirv_GroupCMulINTELjjN5__spv13complex_floatE, CMulINTEL,
    __CLC_COMPLEX_MUL, complex_float, ((complex_float){1, 0}))
__CLC_GROUP_COLLECTIVE_MANUAL_MANGLE(
    _Z22__spirv_GroupCMulINTELjjN5__spv14complex_doubleE, CMulINTEL,
    __CLC_COMPLEX_MUL, complex_double, ((complex_double){1, 0}))

__CLC_GROUP_COLLECTIVE(SMin, __CLC_MIN, char, CHAR_MAX)
__CLC_GROUP_COLLECTIVE(UMin, __CLC_MIN, uchar, UCHAR_MAX)
__CLC_GROUP_COLLECTIVE(SMin, __CLC_MIN, short, SHRT_MAX)
__CLC_GROUP_COLLECTIVE(UMin, __CLC_MIN, ushort, USHRT_MAX)
__CLC_GROUP_COLLECTIVE(SMin, __CLC_MIN, int, INT_MAX)
__CLC_GROUP_COLLECTIVE(UMin, __CLC_MIN, uint, UINT_MAX)
__CLC_GROUP_COLLECTIVE(SMin, __CLC_MIN, long, LONG_MAX)
__CLC_GROUP_COLLECTIVE(UMin, __CLC_MIN, ulong, ULONG_MAX)
__CLC_GROUP_COLLECTIVE(FMin, __CLC_MIN, half, INFINITY)
__CLC_GROUP_COLLECTIVE(FMin, __CLC_MIN, float, INFINITY)
__CLC_GROUP_COLLECTIVE(FMin, __CLC_MIN, double, INFINITY)

__CLC_GROUP_COLLECTIVE(SMax, __CLC_MAX, char, CHAR_MIN)
__CLC_GROUP_COLLECTIVE(UMax, __CLC_MAX, uchar, 0)
__CLC_GROUP_COLLECTIVE(SMax, __CLC_MAX, short, SHRT_MIN)
__CLC_GROUP_COLLECTIVE(UMax, __CLC_MAX, ushort, 0)
__CLC_GROUP_COLLECTIVE(SMax, __CLC_MAX, int, INT_MIN)
__CLC_GROUP_COLLECTIVE(UMax, __CLC_MAX, uint, 0)
__CLC_GROUP_COLLECTIVE(SMax, __CLC_MAX, long, LONG_MIN)
__CLC_GROUP_COLLECTIVE(UMax, __CLC_MAX, ulong, 0)
__CLC_GROUP_COLLECTIVE(FMax, __CLC_MAX, half, -INFINITY)
__CLC_GROUP_COLLECTIVE(FMax, __CLC_MAX, float, -INFINITY)
__CLC_GROUP_COLLECTIVE(FMax, __CLC_MAX, double, -INFINITY)

__CLC_GROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, uchar, ~0)
__CLC_GROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, uchar, 0)
__CLC_GROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, uchar, 0)
__CLC_GROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, char, ~0)
__CLC_GROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, char, 0)
__CLC_GROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, char, 0)

__CLC_GROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, ushort, ~0)
__CLC_GROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, ushort, 0)
__CLC_GROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, ushort, 0)
__CLC_GROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, short, ~0)
__CLC_GROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, short, 0)
__CLC_GROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, short, 0)

__CLC_GROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, uint, ~0)
__CLC_GROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, uint, 0)
__CLC_GROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, uint, 0)
__CLC_GROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, int, ~0)
__CLC_GROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, int, 0)
__CLC_GROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, int, 0)

__CLC_GROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, ulong, ~0l)
__CLC_GROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, ulong, 0l)
__CLC_GROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, ulong, 0l)
__CLC_GROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, long, ~0l)
__CLC_GROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, long, 0l)
__CLC_GROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, long, 0l)

__CLC_GROUP_COLLECTIVE(LogicalOrKHR, __CLC_LOGICAL_OR, bool, false)
__CLC_GROUP_COLLECTIVE(LogicalAndKHR, __CLC_LOGICAL_AND, bool, true)

// half requires additional mangled entry points
#define __CLC_GROUP_COLLECTIVE__DF16(MANGLED_NAME, SPIRV_DISPATCH)             \
  _CLC_DEF _CLC_CONVERGENT half MANGLED_NAME(uint scope, uint op, half x) {    \
    return SPIRV_DISPATCH(scope, op, x);                                       \
  }
__CLC_GROUP_COLLECTIVE__DF16(_Z17__spirv_GroupFAddjjDF16_, __spirv_GroupFAdd)
__CLC_GROUP_COLLECTIVE__DF16(_Z17__spirv_GroupFMinjjDF16_, __spirv_GroupFMin)
__CLC_GROUP_COLLECTIVE__DF16(_Z17__spirv_GroupFMaxjjDF16_, __spirv_GroupFMax)
__CLC_GROUP_COLLECTIVE__DF16(_Z20__spirv_GroupFMulKHRjjDF16_,
                             __spirv_GroupFMulKHR)
#undef __CLC_GROUP_COLLECTIVE__DF16

#undef __CLC_GROUP_COLLECTIVE_4
#undef __CLC_GROUP_COLLECTIVE_5
#undef DISPATCH_TO_CLC_GROUP_COLLECTIVE_MACRO
#undef __CLC_GROUP_COLLECTIVE

#undef __CLC_AND
#undef __CLC_OR
#undef __CLC_MAX
#undef __CLC_MIN
#undef __CLC_ADD
#undef __CLC_MUL

long __clc__get_linear_local_id() {
  size_t id_x = __spirv_LocalInvocationId_x();
  size_t id_y = __spirv_LocalInvocationId_y();
  size_t id_z = __spirv_LocalInvocationId_z();
  size_t size_x = __spirv_WorkgroupSize_x();
  size_t size_y = __spirv_WorkgroupSize_y();
  size_t size_z = __spirv_WorkgroupSize_z();
  uint sg_size = __spirv_SubgroupMaxSize();
  return (id_z * size_y * size_x + id_y * size_x + id_x);
}

long __clc__2d_to_linear_local_id(ulong2 id) {
  size_t size_x = __spirv_WorkgroupSize_x();
  size_t size_y = __spirv_WorkgroupSize_y();
  return (id.y * size_x + id.x);
}

long __clc__3d_to_linear_local_id(ulong3 id) {
  size_t size_x = __spirv_WorkgroupSize_x();
  size_t size_y = __spirv_WorkgroupSize_y();
  size_t size_z = __spirv_WorkgroupSize_z();
  return (id.z * size_y * size_x + id.y * size_x + id.x);
}

#define __CLC_GROUP_BROADCAST(TYPE)                                            \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT TYPE __spirv_GroupBroadcast(          \
      uint scope, TYPE x, ulong local_id) {                                    \
    if (scope == Subgroup) {                                                   \
      return __clc__SubgroupShuffle(x, local_id);                              \
    }                                                                          \
    bool source = (__clc__get_linear_local_id() == local_id);                  \
    __local TYPE *scratch = __CLC_APPEND(__clc__get_group_scratch_, TYPE)();   \
    if (source) {                                                              \
      *scratch = x;                                                            \
    }                                                                          \
    __spirv_ControlBarrier(Workgroup, 0, 0);                                   \
    TYPE result = *scratch;                                                    \
    __spirv_ControlBarrier(Workgroup, 0, 0);                                   \
    return result;                                                             \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT TYPE __spirv_GroupBroadcast(          \
      uint scope, TYPE x, ulong2 local_id) {                                   \
    ulong linear_local_id = __clc__2d_to_linear_local_id(local_id);            \
    return __spirv_GroupBroadcast(scope, x, linear_local_id);                  \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT TYPE __spirv_GroupBroadcast(          \
      uint scope, TYPE x, ulong3 local_id) {                                   \
    ulong linear_local_id = __clc__3d_to_linear_local_id(local_id);            \
    return __spirv_GroupBroadcast(scope, x, linear_local_id);                  \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT TYPE __spirv_GroupBroadcast(          \
      uint scope, TYPE x, uint local_id) {                                     \
    return __spirv_GroupBroadcast(scope, x, (ulong)local_id);                  \
  }
__CLC_GROUP_BROADCAST(char);
__CLC_GROUP_BROADCAST(uchar);
__CLC_GROUP_BROADCAST(short);
__CLC_GROUP_BROADCAST(ushort);
__CLC_GROUP_BROADCAST(int)
__CLC_GROUP_BROADCAST(uint)
__CLC_GROUP_BROADCAST(long)
__CLC_GROUP_BROADCAST(ulong)
__CLC_GROUP_BROADCAST(half)
__CLC_GROUP_BROADCAST(float)
__CLC_GROUP_BROADCAST(double)

// half requires additional mangled entry points
_CLC_DEF _CLC_CONVERGENT half
_Z17__spirv_GroupBroadcastjDF16_m(uint scope, half x, ulong local_id) {
  return __spirv_GroupBroadcast(scope, x, local_id);
}
_CLC_DEF _CLC_CONVERGENT half
_Z17__spirv_GroupBroadcastjDF16_Dv2_m(uint scope, half x, ulong2 local_id) {
  return __spirv_GroupBroadcast(scope, x, local_id);
}
_CLC_DEF _CLC_CONVERGENT half
_Z17__spirv_GroupBroadcastjDF16_Dv3_m(uint scope, half x, ulong3 local_id) {
  return __spirv_GroupBroadcast(scope, x, local_id);
}
_CLC_DEF _CLC_CONVERGENT half _Z22__spirv_GroupBroadcastjDF16_j(uint scope,
                                                                half x,
                                                                uint local_id) {
  return __spirv_GroupBroadcast(scope, x, (ulong)local_id);
}

#undef __CLC_GROUP_BROADCAST

#undef __CLC_APPEND
