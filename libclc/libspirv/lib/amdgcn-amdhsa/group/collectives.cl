//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>
#include <libspirv/spirv_types.h>

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// CLC helpers
__local bool *
__clc__get_group_scratch_bool() __asm("__clc__get_group_scratch_bool");
__local char *
__clc__get_group_scratch_char() __asm("__clc__get_group_scratch_char");
__local schar *
__clc__get_group_scratch_schar() __asm("__clc__get_group_scratch_char");
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

#define __CLC_SUBGROUP_COLLECTIVE_BODY(OP, TYPE, IDENTITY)                     \
  uint sg_lid = __spirv_SubgroupLocalInvocationId();                           \
  /* Can't use XOR/butterfly shuffles; some lanes may be inactive */           \
  for (int o = 1; o < __spirv_SubgroupMaxSize(); o *= 2) {                     \
    TYPE contribution = __spirv_SubgroupShuffleUpINTEL(x, x, o);               \
    bool inactive = (sg_lid < o);                                              \
    contribution = (inactive) ? IDENTITY : contribution;                       \
    x = OP(x, contribution);                                                   \
  }                                                                            \
  /* For Reduce, broadcast result from highest active lane */                  \
  TYPE result;                                                                 \
  if (op == Reduce) {                                                          \
    result = __spirv_SubgroupShuffleINTEL(x, __spirv_SubgroupSize() - 1);      \
    *carry = result;                                                           \
  } /* For InclusiveScan, use results as computed */                           \
  else if (op == InclusiveScan) {                                              \
    result = x;                                                                \
    *carry = result;                                                           \
  } /* For ExclusiveScan, shift and prepend identity */                        \
  else if (op == ExclusiveScan) {                                              \
    *carry = x;                                                                \
    result = __spirv_SubgroupShuffleUpINTEL(x, x, 1);                          \
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

__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, char, 0)
__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, schar, 0)
__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, uchar, 0)
__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, short, 0)
__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, ushort, 0)
__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, int, 0)
__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, uint, 0)
__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, long, 0)
__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, ulong, 0)
__CLC_SUBGROUP_COLLECTIVE(FAdd, __CLC_ADD, half, 0)
__CLC_SUBGROUP_COLLECTIVE(FAdd, __CLC_ADD, float, 0)
__CLC_SUBGROUP_COLLECTIVE(FAdd, __CLC_ADD, double, 0)

__CLC_SUBGROUP_COLLECTIVE(IMulKHR, __CLC_MUL, char, 1)
__CLC_SUBGROUP_COLLECTIVE(IMulKHR, __CLC_MUL, schar, 1)
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

__CLC_SUBGROUP_COLLECTIVE(SMin, __CLC_MIN, char, CHAR_MAX)
__CLC_SUBGROUP_COLLECTIVE(SMin, __CLC_MIN, schar, SCHAR_MAX)
__CLC_SUBGROUP_COLLECTIVE(UMin, __CLC_MIN, uchar, UCHAR_MAX)
__CLC_SUBGROUP_COLLECTIVE(SMin, __CLC_MIN, short, SHRT_MAX)
__CLC_SUBGROUP_COLLECTIVE(UMin, __CLC_MIN, ushort, USHRT_MAX)
__CLC_SUBGROUP_COLLECTIVE(SMin, __CLC_MIN, int, INT_MAX)
__CLC_SUBGROUP_COLLECTIVE(UMin, __CLC_MIN, uint, UINT_MAX)
__CLC_SUBGROUP_COLLECTIVE(SMin, __CLC_MIN, long, LONG_MAX)
__CLC_SUBGROUP_COLLECTIVE(UMin, __CLC_MIN, ulong, ULONG_MAX)
__CLC_SUBGROUP_COLLECTIVE(FMin, __CLC_MIN, half, INFINITY)
__CLC_SUBGROUP_COLLECTIVE(FMin, __CLC_MIN, float, INFINITY)
__CLC_SUBGROUP_COLLECTIVE(FMin, __CLC_MIN, double, INFINITY)

__CLC_SUBGROUP_COLLECTIVE(SMax, __CLC_MAX, char, CHAR_MIN)
__CLC_SUBGROUP_COLLECTIVE(SMax, __CLC_MAX, schar, SCHAR_MIN)
__CLC_SUBGROUP_COLLECTIVE(UMax, __CLC_MAX, uchar, 0)
__CLC_SUBGROUP_COLLECTIVE(SMax, __CLC_MAX, short, SHRT_MIN)
__CLC_SUBGROUP_COLLECTIVE(UMax, __CLC_MAX, ushort, 0)
__CLC_SUBGROUP_COLLECTIVE(SMax, __CLC_MAX, int, INT_MIN)
__CLC_SUBGROUP_COLLECTIVE(UMax, __CLC_MAX, uint, 0)
__CLC_SUBGROUP_COLLECTIVE(SMax, __CLC_MAX, long, LONG_MIN)
__CLC_SUBGROUP_COLLECTIVE(UMax, __CLC_MAX, ulong, 0)
__CLC_SUBGROUP_COLLECTIVE(FMax, __CLC_MAX, half, -INFINITY)
__CLC_SUBGROUP_COLLECTIVE(FMax, __CLC_MAX, float, -INFINITY)
__CLC_SUBGROUP_COLLECTIVE(FMax, __CLC_MAX, double, -INFINITY)

__CLC_SUBGROUP_COLLECTIVE(All, __CLC_AND, bool, true)
__CLC_SUBGROUP_COLLECTIVE(Any, __CLC_OR, bool, false)

__CLC_SUBGROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, uchar, ~0)
__CLC_SUBGROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, uchar, 0)
__CLC_SUBGROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, uchar, 0)
__CLC_SUBGROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, schar, ~0)
__CLC_SUBGROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, schar, 0)
__CLC_SUBGROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, schar, 0)
__CLC_SUBGROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, char, ~0)
__CLC_SUBGROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, char, 0)
__CLC_SUBGROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, char, 0)

__CLC_SUBGROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, ushort, ~0)
__CLC_SUBGROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, ushort, 0)
__CLC_SUBGROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, ushort, 0)
__CLC_SUBGROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, short, ~0)
__CLC_SUBGROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, short, 0)
__CLC_SUBGROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, short, 0)

__CLC_SUBGROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, uint, ~0)
__CLC_SUBGROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, uint, 0)
__CLC_SUBGROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, uint, 0)
__CLC_SUBGROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, int, ~0)
__CLC_SUBGROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, int, 0)
__CLC_SUBGROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, int, 0)

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

#define __CLC_GROUP_COLLECTIVE_INNER(SPIRV_NAME, CLC_NAME, OP, TYPE, IDENTITY) \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT TYPE __CLC_APPEND(                    \
      __spirv_Group, SPIRV_NAME)(int scope, int op, TYPE x) {                \
    TYPE carry = IDENTITY;                                                     \
    /* Perform GroupOperation within sub-group */                              \
    TYPE sg_x = __CLC_APPEND(__clc__Subgroup, CLC_NAME)(op, x, &carry);        \
    if (scope == Subgroup) {                                                   \
      return sg_x;                                                             \
    }                                                                          \
    __local TYPE *scratch = __CLC_APPEND(__clc__get_group_scratch_, TYPE)();   \
    uint sg_id = __spirv_SubgroupId();                                         \
    uint num_sg = __spirv_NumSubgroups();                                      \
    uint sg_lid = __spirv_SubgroupLocalInvocationId();                         \
    uint sg_size = __spirv_SubgroupSize();                                     \
    /* Share carry values across sub-groups */                                 \
    if (sg_lid == sg_size - 1) {                                               \
      scratch[sg_id] = carry;                                                  \
    }                                                                          \
    __spirv_ControlBarrier(Workgroup, Workgroup, AcquireRelease);              \
    /* Perform InclusiveScan over sub-group results */                         \
    TYPE sg_prefix;                                                            \
    TYPE sg_aggregate = scratch[0];                                            \
    for (int s = 1; s < num_sg; ++s) {                                         \
      if (sg_id == s) {                                                        \
        sg_prefix = sg_aggregate;                                              \
      }                                                                        \
      TYPE addend = scratch[s];                                                \
      sg_aggregate = OP(sg_aggregate, addend);                                 \
    }                                                                          \
    /* For Reduce, broadcast result from final sub-group */                    \
    /* For Scan, combine results from previous sub-groups */                   \
    TYPE result;                                                               \
    if (op == Reduce) {                                                        \
      result = sg_aggregate;                                                   \
    } else if (op == InclusiveScan || op == ExclusiveScan) {                   \
      if (sg_id == 0) {                                                        \
        result = sg_x;                                                         \
      } else {                                                                 \
        result = OP(sg_x, sg_prefix);                                          \
      }                                                                        \
    }                                                                          \
    __spirv_ControlBarrier(Workgroup, Workgroup, AcquireRelease);              \
    return result;                                                             \
  }

#define __CLC_GROUP_COLLECTIVE_4(NAME, OP, TYPE, IDENTITY)                     \
  __CLC_GROUP_COLLECTIVE_INNER(NAME, NAME, OP, TYPE, IDENTITY)
#define __CLC_GROUP_COLLECTIVE_5(SPIRV_NAME, CLC_NAME, OP, TYPE, IDENTITY)     \
  __CLC_GROUP_COLLECTIVE_INNER(SPIRV_NAME, CLC_NAME, OP, TYPE, IDENTITY)

#define DISPATCH_TO_CLC_GROUP_COLLECTIVE_MACRO(_1, _2, _3, _4, _5, NAME, ...)  \
  NAME
#define __CLC_GROUP_COLLECTIVE(...)                                            \
  DISPATCH_TO_CLC_GROUP_COLLECTIVE_MACRO(                                      \
      __VA_ARGS__, __CLC_GROUP_COLLECTIVE_5, __CLC_GROUP_COLLECTIVE_4)         \
  (__VA_ARGS__)

__CLC_GROUP_COLLECTIVE(Any, __CLC_OR, bool, false);
__CLC_GROUP_COLLECTIVE(All, __CLC_AND, bool, true);
_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT bool __spirv_GroupAny(int scope,
                                                             bool predicate) {
  return __spirv_GroupAny(scope, Reduce, predicate);
}
_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT bool __spirv_GroupAll(int scope,
                                                             bool predicate) {
  return __spirv_GroupAll(scope, Reduce, predicate);
}

__CLC_GROUP_COLLECTIVE(IAdd, __CLC_ADD, char, 0)
__CLC_GROUP_COLLECTIVE(IAdd, __CLC_ADD, schar, 0)
__CLC_GROUP_COLLECTIVE(IAdd, __CLC_ADD, short, 0)
__CLC_GROUP_COLLECTIVE(IAdd, __CLC_ADD, int, 0)
__CLC_GROUP_COLLECTIVE(IAdd, __CLC_ADD, long, 0)
__CLC_GROUP_COLLECTIVE(FAdd, __CLC_ADD, half, 0)
__CLC_GROUP_COLLECTIVE(FAdd, __CLC_ADD, float, 0)
__CLC_GROUP_COLLECTIVE(FAdd, __CLC_ADD, double, 0)

__CLC_GROUP_COLLECTIVE(IMulKHR, __CLC_MUL, char, 1)
__CLC_GROUP_COLLECTIVE(IMulKHR, __CLC_MUL, schar, 1)
__CLC_GROUP_COLLECTIVE(IMulKHR, __CLC_MUL, short, 1)
__CLC_GROUP_COLLECTIVE(IMulKHR, __CLC_MUL, int, 1)
__CLC_GROUP_COLLECTIVE(IMulKHR, __CLC_MUL, long, 1)
__CLC_GROUP_COLLECTIVE(FMulKHR, __CLC_MUL, half, 1)
__CLC_GROUP_COLLECTIVE(FMulKHR, __CLC_MUL, float, 1)
__CLC_GROUP_COLLECTIVE(FMulKHR, __CLC_MUL, double, 1)

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

__CLC_GROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, schar, ~0)
__CLC_GROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, schar, 0)
__CLC_GROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, schar, 0)
__CLC_GROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, char, ~0)
__CLC_GROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, char, 0)
__CLC_GROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, char, 0)

__CLC_GROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, short, ~0)
__CLC_GROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, short, 0)
__CLC_GROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, short, 0)

__CLC_GROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, int, ~0)
__CLC_GROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, int, 0)
__CLC_GROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, int, 0)

__CLC_GROUP_COLLECTIVE(BitwiseAndKHR, __CLC_AND, long, ~0l)
__CLC_GROUP_COLLECTIVE(BitwiseOrKHR, __CLC_OR, long, 0l)
__CLC_GROUP_COLLECTIVE(BitwiseXorKHR, __CLC_XOR, long, 0l)

__CLC_GROUP_COLLECTIVE(LogicalOrKHR, __CLC_LOGICAL_OR, bool, false)
__CLC_GROUP_COLLECTIVE(LogicalAndKHR, __CLC_LOGICAL_AND, bool, true)

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
      int scope, TYPE x, ulong local_id) {                                     \
    if (scope == Subgroup) {                                                   \
      return __spirv_SubgroupShuffleINTEL(x, local_id);                        \
    }                                                                          \
    bool source = (__spirv_LocalInvocationIndex() == local_id);                \
    __local TYPE *scratch = __CLC_APPEND(__clc__get_group_scratch_, TYPE)();   \
    if (source) {                                                              \
      *scratch = x;                                                            \
    }                                                                          \
    __spirv_ControlBarrier(Workgroup, Workgroup, AcquireRelease);              \
    TYPE result = *scratch;                                                    \
    __spirv_ControlBarrier(Workgroup, Workgroup, AcquireRelease);              \
    return result;                                                             \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT TYPE __spirv_GroupBroadcast(          \
      int scope, TYPE x, ulong2 local_id) {                                    \
    ulong linear_local_id = __clc__2d_to_linear_local_id(local_id);            \
    return __spirv_GroupBroadcast(scope, x, linear_local_id);                  \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT TYPE __spirv_GroupBroadcast(          \
      int scope, TYPE x, ulong3 local_id) {                                    \
    ulong linear_local_id = __clc__3d_to_linear_local_id(local_id);            \
    return __spirv_GroupBroadcast(scope, x, linear_local_id);                  \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT TYPE __spirv_GroupBroadcast(          \
      int scope, TYPE x, uint local_id) {                                      \
    return __spirv_GroupBroadcast(scope, x, (ulong)local_id);                  \
  }
__CLC_GROUP_BROADCAST(char);
__CLC_GROUP_BROADCAST(schar);
__CLC_GROUP_BROADCAST(short);
__CLC_GROUP_BROADCAST(int)
__CLC_GROUP_BROADCAST(long)
__CLC_GROUP_BROADCAST(half)
__CLC_GROUP_BROADCAST(float)
__CLC_GROUP_BROADCAST(double)

#undef __CLC_GROUP_BROADCAST

#undef __CLC_APPEND
