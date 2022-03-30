//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

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

#define __CLC_DECLARE_SHUFFLES(TYPE, TYPE_MANGLED)                             \
  _CLC_DECL TYPE _Z28__spirv_SubgroupShuffleINTELI##TYPE_MANGLED##ET_S0_j(     \
      TYPE, int);                                                              \
  _CLC_DECL TYPE                                                               \
      _Z30__spirv_SubgroupShuffleUpINTELI##TYPE_MANGLED##ET_S0_S0_j(           \
          TYPE, TYPE, unsigned int);

__CLC_DECLARE_SHUFFLES(char, a);
__CLC_DECLARE_SHUFFLES(unsigned char, h);
__CLC_DECLARE_SHUFFLES(short, s);
__CLC_DECLARE_SHUFFLES(unsigned short, t);
__CLC_DECLARE_SHUFFLES(int, i);
__CLC_DECLARE_SHUFFLES(unsigned int, j);
__CLC_DECLARE_SHUFFLES(float, f);
__CLC_DECLARE_SHUFFLES(long, l);
__CLC_DECLARE_SHUFFLES(unsigned long, m);
__CLC_DECLARE_SHUFFLES(double, d);

#undef __CLC_DECLARE_SHUFFLES

#define __CLC_APPEND(NAME, SUFFIX) NAME##SUFFIX

#define __CLC_ADD(x, y) (x + y)
#define __CLC_MIN(x, y) ((x < y) ? (x) : (y))
#define __CLC_MAX(x, y) ((x > y) ? (x) : (y))
#define __CLC_OR(x, y) (x | y)
#define __CLC_AND(x, y) (x & y)
#define __CLC_MUL(x, y) (x * y)

#define __CLC_SUBGROUP_COLLECTIVE_BODY(OP, TYPE, TYPE_MANGLED, IDENTITY)       \
  uint sg_lid = __spirv_SubgroupLocalInvocationId();                           \
  /* Can't use XOR/butterfly shuffles; some lanes may be inactive */           \
  for (int o = 1; o < __spirv_SubgroupMaxSize(); o *= 2) {                     \
    TYPE contribution =                                                        \
        _Z30__spirv_SubgroupShuffleUpINTELI##TYPE_MANGLED##ET_S0_S0_j(x, x,    \
                                                                      o);      \
    bool inactive = (sg_lid < o);                                              \
    contribution = (inactive) ? IDENTITY : contribution;                       \
    x = OP(x, contribution);                                                   \
  }                                                                            \
  /* For Reduce, broadcast result from highest active lane */                  \
  TYPE result;                                                                 \
  if (op == Reduce) {                                                          \
    result = _Z28__spirv_SubgroupShuffleINTELI##TYPE_MANGLED##ET_S0_j(         \
        x, __spirv_SubgroupSize() - 1);                                        \
    *carry = result;                                                           \
  } /* For InclusiveScan, use results as computed */                           \
  else if (op == InclusiveScan) {                                              \
    result = x;                                                                \
    *carry = result;                                                           \
  } /* For ExclusiveScan, shift and prepend identity */                        \
  else if (op == ExclusiveScan) {                                              \
    *carry = x;                                                                \
    result = _Z30__spirv_SubgroupShuffleUpINTELI##TYPE_MANGLED##ET_S0_S0_j(    \
        x, x, 1);                                                              \
    if (sg_lid == 0) {                                                         \
      result = IDENTITY;                                                       \
    }                                                                          \
  }                                                                            \
  return result;

#define __CLC_SUBGROUP_COLLECTIVE(NAME, OP, TYPE, TYPE_MANGLED, IDENTITY)      \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT TYPE __CLC_APPEND(                    \
      __clc__Subgroup, NAME)(uint op, TYPE x, TYPE * carry) {                  \
    __CLC_SUBGROUP_COLLECTIVE_BODY(OP, TYPE, TYPE_MANGLED, IDENTITY)           \
  }

__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, char, a, 0)
__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, uchar, h, 0)
__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, short, s, 0)
__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, ushort, t, 0)
__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, int, i, 0)
__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, uint, j, 0)
__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, long, l, 0)
__CLC_SUBGROUP_COLLECTIVE(IAdd, __CLC_ADD, ulong, m, 0)
__CLC_SUBGROUP_COLLECTIVE(FAdd, __CLC_ADD, float, f, 0)
__CLC_SUBGROUP_COLLECTIVE(FAdd, __CLC_ADD, double, d, 0)

__CLC_SUBGROUP_COLLECTIVE(IMul, __CLC_MUL, char, a, 1)
__CLC_SUBGROUP_COLLECTIVE(IMul, __CLC_MUL, uchar, h, 1)
__CLC_SUBGROUP_COLLECTIVE(IMul, __CLC_MUL, short, s, 1)
__CLC_SUBGROUP_COLLECTIVE(IMul, __CLC_MUL, ushort, t, 1)
__CLC_SUBGROUP_COLLECTIVE(IMul, __CLC_MUL, int, i, 1)
__CLC_SUBGROUP_COLLECTIVE(IMul, __CLC_MUL, uint, j, 1)
__CLC_SUBGROUP_COLLECTIVE(IMul, __CLC_MUL, long, l, 1)
__CLC_SUBGROUP_COLLECTIVE(IMul, __CLC_MUL, ulong, m, 1)
__CLC_SUBGROUP_COLLECTIVE(FMul, __CLC_MUL, float, f, 1)
__CLC_SUBGROUP_COLLECTIVE(FMul, __CLC_MUL, double, d, 1)

__CLC_SUBGROUP_COLLECTIVE(SMin, __CLC_MIN, char, a, CHAR_MAX)
__CLC_SUBGROUP_COLLECTIVE(UMin, __CLC_MIN, uchar, h, UCHAR_MAX)
__CLC_SUBGROUP_COLLECTIVE(SMin, __CLC_MIN, short, s, SHRT_MAX)
__CLC_SUBGROUP_COLLECTIVE(UMin, __CLC_MIN, ushort, t, USHRT_MAX)
__CLC_SUBGROUP_COLLECTIVE(SMin, __CLC_MIN, int, i, INT_MAX)
__CLC_SUBGROUP_COLLECTIVE(UMin, __CLC_MIN, uint, j, UINT_MAX)
__CLC_SUBGROUP_COLLECTIVE(SMin, __CLC_MIN, long, l, LONG_MAX)
__CLC_SUBGROUP_COLLECTIVE(UMin, __CLC_MIN, ulong, m, ULONG_MAX)
__CLC_SUBGROUP_COLLECTIVE(FMin, __CLC_MIN, float, f, FLT_MAX)
__CLC_SUBGROUP_COLLECTIVE(FMin, __CLC_MIN, double, d, DBL_MAX)

__CLC_SUBGROUP_COLLECTIVE(SMax, __CLC_MAX, char, a, CHAR_MIN)
__CLC_SUBGROUP_COLLECTIVE(UMax, __CLC_MAX, uchar, h, 0)
__CLC_SUBGROUP_COLLECTIVE(SMax, __CLC_MAX, short, s, SHRT_MIN)
__CLC_SUBGROUP_COLLECTIVE(UMax, __CLC_MAX, ushort, t, 0)
__CLC_SUBGROUP_COLLECTIVE(SMax, __CLC_MAX, int, i, INT_MIN)
__CLC_SUBGROUP_COLLECTIVE(UMax, __CLC_MAX, uint, j, 0)
__CLC_SUBGROUP_COLLECTIVE(SMax, __CLC_MAX, long, l, LONG_MIN)
__CLC_SUBGROUP_COLLECTIVE(UMax, __CLC_MAX, ulong, m, 0)
__CLC_SUBGROUP_COLLECTIVE(FMax, __CLC_MAX, float, f, -FLT_MAX)
__CLC_SUBGROUP_COLLECTIVE(FMax, __CLC_MAX, double, d, -DBL_MAX)

__CLC_SUBGROUP_COLLECTIVE(All, __CLC_AND, bool, a, true)
__CLC_SUBGROUP_COLLECTIVE(Any, __CLC_OR, bool, a, true)

#undef __CLC_SUBGROUP_COLLECTIVE_BODY
#undef __CLC_SUBGROUP_COLLECTIVE

#define __CLC_GROUP_COLLECTIVE_INNER(SPIRV_NAME, CLC_NAME, OP, TYPE, IDENTITY) \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT TYPE __CLC_APPEND(                    \
      __spirv_Group, SPIRV_NAME)(uint scope, uint op, TYPE x) {                \
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
    __spirv_ControlBarrier(Workgroup, 0, 0);                                   \
    /* Perform InclusiveScan over sub-group results */                         \
    TYPE sg_prefix;                                                            \
    TYPE sg_aggregate = scratch[0];                                            \
    _Pragma("unroll") for (int s = 1; s < num_sg; ++s) {                       \
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
    __spirv_ControlBarrier(Workgroup, 0, 0);                                   \
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
_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT bool __spirv_GroupAny(uint scope,
                                                             bool predicate) {
  return __spirv_GroupAny(scope, Reduce, predicate);
}
_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT bool __spirv_GroupAll(uint scope,
                                                             bool predicate) {
  return __spirv_GroupAll(scope, Reduce, predicate);
}

__CLC_GROUP_COLLECTIVE(IAdd, __CLC_ADD, char, 0)
__CLC_GROUP_COLLECTIVE(IAdd, __CLC_ADD, uchar, 0)
__CLC_GROUP_COLLECTIVE(IAdd, __CLC_ADD, short, 0)
__CLC_GROUP_COLLECTIVE(IAdd, __CLC_ADD, ushort, 0)
__CLC_GROUP_COLLECTIVE(IAdd, __CLC_ADD, int, 0)
__CLC_GROUP_COLLECTIVE(IAdd, __CLC_ADD, uint, 0)
__CLC_GROUP_COLLECTIVE(IAdd, __CLC_ADD, long, 0)
__CLC_GROUP_COLLECTIVE(IAdd, __CLC_ADD, ulong, 0)
__CLC_GROUP_COLLECTIVE(FAdd, __CLC_ADD, float, 0)
__CLC_GROUP_COLLECTIVE(FAdd, __CLC_ADD, double, 0)

// There is no Mul group op in SPIR-V, use non-uniform variant instead.
__CLC_GROUP_COLLECTIVE(NonUniformIMul, IMul, __CLC_MUL, char, 1)
__CLC_GROUP_COLLECTIVE(NonUniformIMul, IMul, __CLC_MUL, uchar, 1)
__CLC_GROUP_COLLECTIVE(NonUniformIMul, IMul, __CLC_MUL, short, 1)
__CLC_GROUP_COLLECTIVE(NonUniformIMul, IMul, __CLC_MUL, ushort, 1)
__CLC_GROUP_COLLECTIVE(NonUniformIMul, IMul, __CLC_MUL, int, 1)
__CLC_GROUP_COLLECTIVE(NonUniformIMul, IMul, __CLC_MUL, uint, 1)
__CLC_GROUP_COLLECTIVE(NonUniformIMul, IMul, __CLC_MUL, long, 1)
__CLC_GROUP_COLLECTIVE(NonUniformIMul, IMul, __CLC_MUL, ulong, 1)
__CLC_GROUP_COLLECTIVE(NonUniformFMul, FMul, __CLC_MUL, float, 1)
__CLC_GROUP_COLLECTIVE(NonUniformFMul, FMul, __CLC_MUL, double, 1)

__CLC_GROUP_COLLECTIVE(SMin, __CLC_MIN, char, CHAR_MAX)
__CLC_GROUP_COLLECTIVE(UMin, __CLC_MIN, uchar, UCHAR_MAX)
__CLC_GROUP_COLLECTIVE(SMin, __CLC_MIN, short, SHRT_MAX)
__CLC_GROUP_COLLECTIVE(UMin, __CLC_MIN, ushort, USHRT_MAX)
__CLC_GROUP_COLLECTIVE(SMin, __CLC_MIN, int, INT_MAX)
__CLC_GROUP_COLLECTIVE(UMin, __CLC_MIN, uint, UINT_MAX)
__CLC_GROUP_COLLECTIVE(SMin, __CLC_MIN, long, LONG_MAX)
__CLC_GROUP_COLLECTIVE(UMin, __CLC_MIN, ulong, ULONG_MAX)
__CLC_GROUP_COLLECTIVE(FMin, __CLC_MIN, float, FLT_MAX)
__CLC_GROUP_COLLECTIVE(FMin, __CLC_MIN, double, DBL_MAX)

__CLC_GROUP_COLLECTIVE(SMax, __CLC_MAX, char, CHAR_MIN)
__CLC_GROUP_COLLECTIVE(UMax, __CLC_MAX, uchar, 0)
__CLC_GROUP_COLLECTIVE(SMax, __CLC_MAX, short, SHRT_MIN)
__CLC_GROUP_COLLECTIVE(UMax, __CLC_MAX, ushort, 0)
__CLC_GROUP_COLLECTIVE(SMax, __CLC_MAX, int, INT_MIN)
__CLC_GROUP_COLLECTIVE(UMax, __CLC_MAX, uint, 0)
__CLC_GROUP_COLLECTIVE(SMax, __CLC_MAX, long, LONG_MIN)
__CLC_GROUP_COLLECTIVE(UMax, __CLC_MAX, ulong, 0)
__CLC_GROUP_COLLECTIVE(FMax, __CLC_MAX, float, -FLT_MAX)
__CLC_GROUP_COLLECTIVE(FMax, __CLC_MAX, double, -DBL_MAX)

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

#define __CLC_GROUP_BROADCAST(TYPE, TYPE_MANGLED)                              \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT TYPE __spirv_GroupBroadcast(          \
      uint scope, TYPE x, ulong local_id) {                                    \
    if (scope == Subgroup) {                                                   \
      return _Z28__spirv_SubgroupShuffleINTELI##TYPE_MANGLED##ET_S0_j(         \
          x, local_id);                                                        \
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
__CLC_GROUP_BROADCAST(char, a);
__CLC_GROUP_BROADCAST(uchar, h);
__CLC_GROUP_BROADCAST(short, s);
__CLC_GROUP_BROADCAST(ushort, t);
__CLC_GROUP_BROADCAST(int, i)
__CLC_GROUP_BROADCAST(uint, j)
__CLC_GROUP_BROADCAST(long, l)
__CLC_GROUP_BROADCAST(ulong, m)
__CLC_GROUP_BROADCAST(float, f)
__CLC_GROUP_BROADCAST(double, d)

#undef __CLC_GROUP_BROADCAST

#undef __CLC_APPEND
