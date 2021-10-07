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

int __nvvm_reflect(const char __constant *);

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

_CLC_DEF _CLC_CONVERGENT uint __clc__membermask() {
  uint FULL_MASK = 0xFFFFFFFF;
  uint max_size = __spirv_SubgroupMaxSize();
  uint sg_size = __spirv_SubgroupSize();
  return FULL_MASK >> (max_size - sg_size);
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
#define __CLC_AND(x, y) (x & y)
#define __CLC_MUL(x, y) (x * y)

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
    if (__nvvm_reflect("__CUDA_ARCH") >= 800 && op == Reduce) {                \
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

__CLC_SUBGROUP_COLLECTIVE(IMul, __CLC_MUL, char, 1)
__CLC_SUBGROUP_COLLECTIVE(IMul, __CLC_MUL, uchar, 1)
__CLC_SUBGROUP_COLLECTIVE(IMul, __CLC_MUL, short, 1)
__CLC_SUBGROUP_COLLECTIVE(IMul, __CLC_MUL, ushort, 1)
__CLC_SUBGROUP_COLLECTIVE(IMul, __CLC_MUL, int, 1)
__CLC_SUBGROUP_COLLECTIVE(IMul, __CLC_MUL, uint, 1)
__CLC_SUBGROUP_COLLECTIVE(IMul, __CLC_MUL, long, 1)
__CLC_SUBGROUP_COLLECTIVE(IMul, __CLC_MUL, ulong, 1)
__CLC_SUBGROUP_COLLECTIVE(FMul, __CLC_MUL, half, 1)
__CLC_SUBGROUP_COLLECTIVE(FMul, __CLC_MUL, float, 1)
__CLC_SUBGROUP_COLLECTIVE(FMul, __CLC_MUL, double, 1)

__CLC_SUBGROUP_COLLECTIVE(SMin, __CLC_MIN, char, CHAR_MAX)
__CLC_SUBGROUP_COLLECTIVE(UMin, __CLC_MIN, uchar, UCHAR_MAX)
__CLC_SUBGROUP_COLLECTIVE(SMin, __CLC_MIN, short, SHRT_MAX)
__CLC_SUBGROUP_COLLECTIVE(UMin, __CLC_MIN, ushort, USHRT_MAX)
__CLC_SUBGROUP_COLLECTIVE_REDUX(SMin, __CLC_MIN, min, int, INT_MAX)
__CLC_SUBGROUP_COLLECTIVE_REDUX(UMin, __CLC_MIN, umin, uint, UINT_MAX)
__CLC_SUBGROUP_COLLECTIVE(SMin, __CLC_MIN, long, LONG_MAX)
__CLC_SUBGROUP_COLLECTIVE(UMin, __CLC_MIN, ulong, ULONG_MAX)
__CLC_SUBGROUP_COLLECTIVE(FMin, __CLC_MIN, half, HALF_MAX)
__CLC_SUBGROUP_COLLECTIVE(FMin, __CLC_MIN, float, FLT_MAX)
__CLC_SUBGROUP_COLLECTIVE(FMin, __CLC_MIN, double, DBL_MAX)

__CLC_SUBGROUP_COLLECTIVE(SMax, __CLC_MAX, char, CHAR_MIN)
__CLC_SUBGROUP_COLLECTIVE(UMax, __CLC_MAX, uchar, 0)
__CLC_SUBGROUP_COLLECTIVE(SMax, __CLC_MAX, short, SHRT_MIN)
__CLC_SUBGROUP_COLLECTIVE(UMax, __CLC_MAX, ushort, 0)
__CLC_SUBGROUP_COLLECTIVE_REDUX(SMax, __CLC_MAX, max, int, INT_MIN)
__CLC_SUBGROUP_COLLECTIVE_REDUX(UMax, __CLC_MAX, umax, uint, 0)
__CLC_SUBGROUP_COLLECTIVE(SMax, __CLC_MAX, long, LONG_MIN)
__CLC_SUBGROUP_COLLECTIVE(UMax, __CLC_MAX, ulong, 0)
__CLC_SUBGROUP_COLLECTIVE(FMax, __CLC_MAX, half, -HALF_MAX)
__CLC_SUBGROUP_COLLECTIVE(FMax, __CLC_MAX, float, -FLT_MAX)
__CLC_SUBGROUP_COLLECTIVE(FMax, __CLC_MAX, double, -DBL_MAX)

#undef __CLC_SUBGROUP_COLLECTIVE_BODY
#undef __CLC_SUBGROUP_COLLECTIVE
#undef __CLC_SUBGROUP_COLLECTIVE_REDUX

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

// There is no Mul group op in SPIR-V, use non-uniform variant instead.
__CLC_GROUP_COLLECTIVE(NonUniformIMul, IMul, __CLC_MUL, char, 1)
__CLC_GROUP_COLLECTIVE(NonUniformIMul, IMul, __CLC_MUL, uchar, 1)
__CLC_GROUP_COLLECTIVE(NonUniformIMul, IMul, __CLC_MUL, short, 1)
__CLC_GROUP_COLLECTIVE(NonUniformIMul, IMul, __CLC_MUL, ushort, 1)
__CLC_GROUP_COLLECTIVE(NonUniformIMul, IMul, __CLC_MUL, int, 1)
__CLC_GROUP_COLLECTIVE(NonUniformIMul, IMul, __CLC_MUL, uint, 1)
__CLC_GROUP_COLLECTIVE(NonUniformIMul, IMul, __CLC_MUL, long, 1)
__CLC_GROUP_COLLECTIVE(NonUniformIMul, IMul, __CLC_MUL, ulong, 1)
__CLC_GROUP_COLLECTIVE(NonUniformFMul, FMul, __CLC_MUL, half, 1)
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
__CLC_GROUP_COLLECTIVE(FMin, __CLC_MIN, half, HALF_MAX)
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
__CLC_GROUP_COLLECTIVE(FMax, __CLC_MAX, half, -HALF_MAX)
__CLC_GROUP_COLLECTIVE(FMax, __CLC_MAX, float, -FLT_MAX)
__CLC_GROUP_COLLECTIVE(FMax, __CLC_MAX, double, -DBL_MAX)

// half requires additional mangled entry points
_CLC_DECL _CLC_CONVERGENT half _Z17__spirv_GroupFAddjjDF16_(uint scope, uint op,
                                                            half x) {
  return __spirv_GroupFAdd(scope, op, x);
}
_CLC_DECL _CLC_CONVERGENT half _Z17__spirv_GroupFMinjjDF16_(uint scope, uint op,
                                                            half x) {
  return __spirv_GroupFMin(scope, op, x);
}
_CLC_DECL _CLC_CONVERGENT half _Z17__spirv_GroupFMaxjjDF16_(uint scope, uint op,
                                                            half x) {
  return __spirv_GroupFMax(scope, op, x);
}

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
_CLC_DECL _CLC_CONVERGENT half
_Z17__spirv_GroupBroadcastjDF16_m(uint scope, half x, ulong local_id) {
  return __spirv_GroupBroadcast(scope, x, local_id);
}
_CLC_DECL _CLC_CONVERGENT half
_Z17__spirv_GroupBroadcastjDF16_Dv2_m(uint scope, half x, ulong2 local_id) {
  return __spirv_GroupBroadcast(scope, x, local_id);
}
_CLC_DECL _CLC_CONVERGENT half
_Z17__spirv_GroupBroadcastjDF16_Dv3_m(uint scope, half x, ulong3 local_id) {
  return __spirv_GroupBroadcast(scope, x, local_id);
}
_CLC_DECL _CLC_CONVERGENT half
_Z22__spirv_GroupBroadcastjDF16_j(uint scope, half x, uint local_id) {
  return __spirv_GroupBroadcast(scope, x, (ulong)local_id);
}

#undef __CLC_GROUP_BROADCAST

#undef __CLC_APPEND
