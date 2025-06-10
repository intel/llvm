//==--- nativecpu_utils.cpp - builtins for Native CPU ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file contains all device builtins and types that are either directly
// referenced in the SYCL source code (e.g. __spirv_...) or used by the
// NativeCPU passes. This is currently built as a device code library but
// could perhaps eventually be compiled as a standard C++ translation unit
// without SYCL attributes etc.

#if defined(__SYCL_NATIVE_CPU__)

#include "device.h"
#include <cstdint>
#include <sycl/__spirv/spirv_ops.hpp>
#include <sycl/vector.hpp>

// including state definition from Native CPU UR adapter
#include "nativecpu_state.hpp"
using __nativecpu_state = native_cpu::state;

#undef DEVICE_EXTERNAL
#undef DEVICE_EXTERN_C
#define DEVICE_EXTERN_C extern "C" SYCL_EXTERNAL
#define DEVICE_EXTERNAL_C DEVICE_EXTERN_C __attribute__((always_inline))
#define DEVICE_EXTERNAL SYCL_EXTERNAL __attribute__((always_inline))

// Several functions are used implicitly by WorkItemLoopsPass and
// PrepareSYCLNativeCPUPass and need to be marked as used to prevent them being
// removed early.
#define USED __attribute__((used))

#define OCL_LOCAL __attribute__((opencl_local))
#define OCL_GLOBAL __attribute__((opencl_global))
#define OCL_PRIVATE __attribute__((opencl_private))

DEVICE_EXTERN_C void __mux_work_group_barrier(int32_t id, int32_t scope,
                                              int32_t semantics) noexcept;
__SYCL_CONVERGENT__ DEVICE_EXTERNAL void
__spirv_ControlBarrier(int32_t Execution, int32_t Memory,
                       int32_t Semantics) noexcept {
  if (__spv::Scope::Flag::Workgroup == Execution)
    // todo: check id and args; use mux constants
    __mux_work_group_barrier(0, Execution, Semantics);
}

DEVICE_EXTERN_C void __mux_mem_barrier(int32_t scope,
                                       int32_t semantics) noexcept;
__SYCL_CONVERGENT__ DEVICE_EXTERNAL void
__spirv_MemoryBarrier(int32_t Memory, int32_t Semantics) noexcept {
  __mux_mem_barrier(Memory, Semantics);
}

#define DefGenericCastToPtrExplImpl(sfx, asp, cv)                              \
  DEVICE_EXTERNAL cv asp void *__spirv_GenericCastToPtrExplicit_##sfx(         \
      cv void *p, int) noexcept {                                              \
    return (cv asp void *)p;                                                   \
  }                                                                            \
  static_assert(true)

#define DefGenericCastToPtrExpl(sfx, asp)                                      \
  DefGenericCastToPtrExplImpl(sfx, asp, );                                     \
  DefGenericCastToPtrExplImpl(sfx, asp, const);                                \
  DefGenericCastToPtrExplImpl(sfx, asp, volatile);                             \
  DefGenericCastToPtrExplImpl(sfx, asp, const volatile)

DefGenericCastToPtrExpl(ToPrivate, OCL_PRIVATE);
DefGenericCastToPtrExpl(ToLocal, OCL_LOCAL);
DefGenericCastToPtrExpl(ToGlobal, OCL_GLOBAL);

#define DefSubgroupBlockINTEL1(Type, PType)                                    \
  template <>                                                                  \
  __SYCL_CONVERGENT__ DEVICE_EXTERNAL Type                                     \
  __spirv_SubgroupBlockReadINTEL<Type>(const OCL_GLOBAL PType *Ptr) noexcept { \
    return Ptr[__spirv_SubgroupLocalInvocationId()];                           \
  }                                                                            \
  __SYCL_CONVERGENT__ DEVICE_EXTERNAL void __spirv_SubgroupBlockWriteINTEL(    \
      PType OCL_GLOBAL *ptr, Type v) noexcept {                                \
    ((Type *)ptr)[__spirv_SubgroupLocalInvocationId()] = v;                    \
  }                                                                            \
  static_assert(true)

#define DefSubgroupBlockINTEL_vt(Type, VT_name)                                \
  DefSubgroupBlockINTEL1(ncpu_types::vtypes<Type>::VT_name, Type)

#define DefSubgroupBlockINTEL(Type)                                            \
  DefSubgroupBlockINTEL1(Type, Type);                                          \
  DefSubgroupBlockINTEL_vt(Type, v2);                                          \
  DefSubgroupBlockINTEL_vt(Type, v4);                                          \
  DefSubgroupBlockINTEL_vt(Type, v8)

namespace ncpu_types {
template <class T> struct vtypes {
  using v2 = typename sycl::vec<T, 2>::vector_t;
  using v4 = typename sycl::vec<T, 4>::vector_t;
  using v8 = typename sycl::vec<T, 8>::vector_t;
};
} // namespace ncpu_types

DefSubgroupBlockINTEL(uint32_t);
DefSubgroupBlockINTEL(uint64_t);
DefSubgroupBlockINTEL(uint8_t);
DefSubgroupBlockINTEL(uint16_t);

#define DefineGOp1(spir_sfx, name)                                             \
  DEVICE_EXTERN_C bool __mux_sub_group_##name##_i1(bool) noexcept;             \
  DEVICE_EXTERN_C bool __mux_work_group_##name##_i1(uint32_t id,               \
                                                    bool val) noexcept;        \
  DEVICE_EXTERNAL bool __spirv_Group##spir_sfx(int32_t g, bool val) noexcept { \
    if (__spv::Scope::Flag::Subgroup == g)                                     \
      return __mux_sub_group_##name##_i1(val);                                 \
    else if (__spv::Scope::Flag::Workgroup == g)                               \
      return __mux_work_group_##name##_i1(0, val);                             \
    return false;                                                              \
  }                                                                            \
  static_assert(true)

DefineGOp1(Any, any);
DefineGOp1(All, all);

#define DefineGOp(Type, MuxType, spir_sfx, mux_sfx)                            \
  DEVICE_EXTERN_C MuxType __mux_sub_group_scan_inclusive_##mux_sfx(            \
      MuxType) noexcept;                                                       \
  DEVICE_EXTERN_C MuxType __mux_sub_group_scan_exclusive_##mux_sfx(            \
      MuxType) noexcept;                                                       \
  DEVICE_EXTERN_C MuxType __mux_sub_group_reduce_##mux_sfx(MuxType) noexcept;  \
  DEVICE_EXTERN_C MuxType __mux_work_group_scan_exclusive_##mux_sfx(           \
      uint32_t, MuxType) noexcept;                                             \
  DEVICE_EXTERN_C MuxType __mux_work_group_scan_inclusive_##mux_sfx(           \
      uint32_t, MuxType) noexcept;                                             \
  DEVICE_EXTERN_C MuxType __mux_work_group_reduce_##mux_sfx(uint32_t,          \
                                                            MuxType) noexcept; \
  DEVICE_EXTERNAL Type __spirv_Group##spir_sfx(int32_t g, int32_t id,          \
                                               Type v) noexcept {              \
    if (__spv::Scope::Flag::Subgroup == g) {                                   \
      if (static_cast<unsigned>(__spv::GroupOperation::InclusiveScan) == id)   \
        return __mux_sub_group_scan_inclusive_##mux_sfx(v);                    \
      if (static_cast<unsigned>(__spv::GroupOperation::ExclusiveScan) == id)   \
        return __mux_sub_group_scan_exclusive_##mux_sfx(v);                    \
      if (static_cast<unsigned>(__spv::GroupOperation::Reduce) == id)          \
        return __mux_sub_group_reduce_##mux_sfx(v);                            \
    } else if (__spv::Scope::Flag::Workgroup == g) {                           \
      uint32_t bid = 0;                                                        \
      if (static_cast<unsigned>(__spv::GroupOperation::ExclusiveScan) == id)   \
        return __mux_work_group_scan_exclusive_##mux_sfx(bid, v);              \
      if (static_cast<unsigned>(__spv::GroupOperation::InclusiveScan) == id)   \
        return __mux_work_group_scan_inclusive_##mux_sfx(bid, v);              \
      if (static_cast<unsigned>(__spv::GroupOperation::Reduce) == id)          \
        return __mux_work_group_reduce_##mux_sfx(bid, v);                      \
    }                                                                          \
    return Type(); /*todo: add support for other flags as they are tested*/    \
  }                                                                            \
  static_assert(true)

#define DefineSignedGOp(Name, MuxName, Bits)                                   \
  DefineGOp(int##Bits##_t, int##Bits##_t, Name, MuxName##Bits)

#define DefineUnsignedGOp(Name, MuxName, Bits)                                 \
  DefineGOp(uint##Bits##_t, int##Bits##_t, Name, MuxName##Bits)

#define Define_32_64(Define, Name, MuxName)                                    \
  Define(Name, MuxName, 32);                                                   \
  Define(Name, MuxName, 64)

// todo: add support for other integer and float types once there are tests
#define DefineIntGOps(Name, MuxName)                                           \
  Define_32_64(DefineSignedGOp, Name, MuxName);                                \
  Define_32_64(DefineUnsignedGOp, Name, MuxName)

#define DefineFPGOps(Name, MuxName)                                            \
  DefineGOp(float, float, Name, MuxName##32);                                  \
  DefineGOp(_Float16, _Float16, Name, MuxName##16);                            \
  DefineGOp(double, double, Name, MuxName##64)

DefineIntGOps(IAdd, add_i);
DefineIntGOps(IMulKHR, mul_i);

Define_32_64(DefineUnsignedGOp, UMin, umin_i);
Define_32_64(DefineUnsignedGOp, UMax, umax_i);
Define_32_64(DefineSignedGOp, SMin, smin_i);
Define_32_64(DefineSignedGOp, SMax, smax_i);

DefineFPGOps(FMulKHR, fmul_f);
DefineFPGOps(FAdd, fadd_f);
DefineFPGOps(FMin, fmin_f);
DefineFPGOps(FMax, fmax_f);

#define DefineBitwiseGroupOp(Type, MuxType, mux_sfx)                           \
  DefineGOp(Type, MuxType, BitwiseOrKHR, or_##mux_sfx);                        \
  DefineGOp(Type, MuxType, BitwiseXorKHR, xor_##mux_sfx);                      \
  DefineGOp(Type, MuxType, BitwiseAndKHR, and_##mux_sfx)

DefineBitwiseGroupOp(int32_t, int32_t, i32);
DefineBitwiseGroupOp(uint32_t, int32_t, i32);
DefineBitwiseGroupOp(int64_t, int64_t, i64);
DefineBitwiseGroupOp(uint64_t, int64_t, i64);

#define DefineLogicalGroupOp(Type, MuxType, mux_sfx)                           \
  DefineGOp(Type, MuxType, LogicalOrKHR, logical_or_##mux_sfx);                \
  DefineGOp(Type, MuxType, LogicalXorKHR, logical_xor_##mux_sfx);              \
  DefineGOp(Type, MuxType, LogicalAndKHR, logical_and_##mux_sfx)

DefineLogicalGroupOp(bool, bool, i1);

#define DefineBroadcastMuxType(Type, Sfx, MuxType, IDType)                     \
  DEVICE_EXTERN_C MuxType __mux_work_group_broadcast_##Sfx(                    \
      int32_t id, MuxType val, uint64_t lidx, uint64_t lidy,                   \
      uint64_t lidz) noexcept;                                                 \
  DEVICE_EXTERN_C MuxType __mux_sub_group_broadcast_##Sfx(                     \
      MuxType val, int32_t sg_lid) noexcept

#define DefineBroadCastImpl(Type, Sfx, MuxType, IDType)                        \
  DEVICE_EXTERNAL Type __spirv_GroupBroadcast(int32_t g, Type v,               \
                                              IDType l) noexcept {             \
    if (__spv::Scope::Flag::Subgroup == g)                                     \
      return __mux_sub_group_broadcast_##Sfx(v, l);                            \
    else                                                                       \
      return __mux_work_group_broadcast_##Sfx(0, v, l, 0, 0);                  \
  }                                                                            \
                                                                               \
  DEVICE_EXTERNAL Type __spirv_GroupBroadcast(                                 \
      int32_t g, Type v, sycl::vec<IDType, 2>::vector_t l) noexcept {          \
    if (__spv::Scope::Flag::Subgroup == g)                                     \
      return __mux_sub_group_broadcast_##Sfx(v, l[0]);                         \
    else                                                                       \
      return __mux_work_group_broadcast_##Sfx(0, v, l[0], l[1], 0);            \
  }                                                                            \
                                                                               \
  DEVICE_EXTERNAL Type __spirv_GroupBroadcast(                                 \
      int32_t g, Type v, sycl::vec<IDType, 3>::vector_t l) noexcept {          \
    if (__spv::Scope::Flag::Subgroup == g)                                     \
      return __mux_sub_group_broadcast_##Sfx(v, l[0]);                         \
    else                                                                       \
      return __mux_work_group_broadcast_##Sfx(0, v, l[0], l[1], l[2]);         \
  }                                                                            \
  static_assert(true)

#define DefineBroadCast(Type, Sfx, MuxType)                                    \
  DefineBroadcastMuxType(Type, Sfx, MuxType, uint32_t);                        \
  DefineBroadcastMuxType(Type, Sfx, MuxType, uint64_t);                        \
  DefineBroadCastImpl(Type, Sfx, MuxType, uint32_t);                           \
  DefineBroadCastImpl(Type, Sfx, MuxType, uint64_t)

DefineBroadCast(uint32_t, i32, int32_t);
DefineBroadCast(int32_t, i32, int32_t);
DefineBroadCast(float, f32, float);
DefineBroadCast(double, f64, double);
DefineBroadCast(uint64_t, i64, int64_t);
DefineBroadCast(int64_t, i64, int64_t);

#define DefShuffleINTEL(Type, Sfx, MuxType)                                    \
  DEVICE_EXTERN_C MuxType __mux_sub_group_shuffle_##Sfx(MuxType val,           \
                                                        int32_t lid) noexcept; \
  DEVICE_EXTERNAL Type __spirv_SubgroupShuffleINTEL(Type val,                  \
                                                    unsigned id) noexcept {    \
    return (Type)__mux_sub_group_shuffle_##Sfx((MuxType)val, id);              \
  }                                                                            \
  static_assert(true)

#define DefShuffleUpINTEL(Type, Sfx, MuxType)                                  \
  DEVICE_EXTERN_C MuxType __mux_sub_group_shuffle_up_##Sfx(                    \
      MuxType prev, MuxType curr, int32_t delta) noexcept;                     \
  DEVICE_EXTERNAL Type __spirv_SubgroupShuffleUpINTEL(                         \
      Type prev, Type curr, unsigned delta) noexcept {                         \
    return (Type)__mux_sub_group_shuffle_up_##Sfx((MuxType)prev,               \
                                                  (MuxType)curr, delta);       \
  }                                                                            \
  static_assert(true)

#define DefShuffleDownINTEL(Type, Sfx, MuxType)                                \
  DEVICE_EXTERN_C MuxType __mux_sub_group_shuffle_down_##Sfx(                  \
      MuxType curr, MuxType next, int32_t delta) noexcept;                     \
  DEVICE_EXTERNAL Type __spirv_SubgroupShuffleDownINTEL(                       \
      Type curr, Type next, unsigned delta) noexcept {                         \
    return (Type)__mux_sub_group_shuffle_down_##Sfx((MuxType)curr,             \
                                                    (MuxType)next, delta);     \
  }                                                                            \
  static_assert(true)

#define DefShuffleXorINTEL(Type, Sfx, MuxType)                                 \
  DEVICE_EXTERN_C MuxType __mux_sub_group_shuffle_xor_##Sfx(MuxType val,       \
                                                            int32_t xor_val);  \
  DEVICE_EXTERNAL Type __spirv_SubgroupShuffleXorINTEL(                        \
      Type data, unsigned value) noexcept {                                    \
    return (Type)__mux_sub_group_shuffle_xor_##Sfx((MuxType)data, value);      \
  }                                                                            \
  static_assert(true)

#define DefShuffleINTEL_All(Type, Sfx, MuxType)                                \
  DefShuffleINTEL(Type, Sfx, MuxType);                                         \
  DefShuffleUpINTEL(Type, Sfx, MuxType);                                       \
  DefShuffleDownINTEL(Type, Sfx, MuxType);                                     \
  DefShuffleXorINTEL(Type, Sfx, MuxType)

DefShuffleINTEL_All(uint64_t, i64, int64_t);
DefShuffleINTEL_All(int64_t, i64, int64_t);
DefShuffleINTEL_All(int32_t, i32, int32_t);
DefShuffleINTEL_All(uint32_t, i32, int32_t);
DefShuffleINTEL_All(int16_t, i16, int16_t);
DefShuffleINTEL_All(uint16_t, i16, int16_t);
DefShuffleINTEL_All(int8_t, i8, int8_t);
DefShuffleINTEL_All(uint8_t, i8, int8_t);
DefShuffleINTEL_All(double, f64, double);
DefShuffleINTEL_All(float, f32, float);
DefShuffleINTEL_All(_Float16, f16, _Float16);

#define DefineShuffleVec(T, N, Sfx, MuxType)                                   \
  using vt##T##N = sycl::vec<T, N>::vector_t;                                  \
  using vt##MuxType##N = sycl::vec<MuxType, N>::vector_t;                      \
  DefShuffleINTEL_All(vt##T##N, v##N##Sfx, vt##MuxType##N)

#define DefineShuffleVec2to16(Type, Sfx, MuxType)                              \
  DefineShuffleVec(Type, 2, Sfx, MuxType);                                     \
  DefineShuffleVec(Type, 4, Sfx, MuxType);                                     \
  DefineShuffleVec(Type, 8, Sfx, MuxType);                                     \
  DefineShuffleVec(Type, 16, Sfx, MuxType)

DefineShuffleVec2to16(int32_t, i32, int32_t);
DefineShuffleVec2to16(uint32_t, i32, int32_t);
DefineShuffleVec2to16(float, f32, float);

#define GET_PROPS __attribute__((pure))
#define GEN_u32(bname, muxname)                                                \
  DEVICE_EXTERN_C GET_PROPS uint32_t muxname();                                \
  DEVICE_EXTERNAL GET_PROPS uint32_t bname() { return muxname(); }             \
  static_assert(true)
// subgroup
GEN_u32(__spirv_SubgroupLocalInvocationId, __mux_get_sub_group_local_id);
GEN_u32(__spirv_SubgroupMaxSize, __mux_get_max_sub_group_size);
GEN_u32(__spirv_SubgroupId, __mux_get_sub_group_id);

// I64_I32
#define GEN_p(bname, muxname, arg)                                             \
  DEVICE_EXTERN_C GET_PROPS uint64_t muxname(uint32_t);                        \
  DEVICE_EXTERNAL GET_PROPS uint64_t bname() { return muxname(arg); }          \
  static_assert(true)

#define GEN_xyz(bname, ncpu_name)                                              \
  GEN_p(bname##_x, ncpu_name, 0);                                              \
  GEN_p(bname##_y, ncpu_name, 1);                                              \
  GEN_p(bname##_z, ncpu_name, 2)

GEN_xyz(__spirv_GlobalOffset, __mux_get_global_offset);
GEN_xyz(__spirv_LocalInvocationId, __mux_get_local_id);
GEN_xyz(__spirv_NumWorkgroups, __mux_get_num_groups);
GEN_xyz(__spirv_WorkgroupSize, __mux_get_local_size);
GEN_xyz(__spirv_WorkgroupId, __mux_get_group_id);

template <class T>
using MakeGlobalType = typename sycl::detail::DecoratedType<
    T, sycl::access::address_space::global_space>::type;

#define DefStateSetWithType(name, field, type)                                 \
  DEVICE_EXTERNAL_C USED void __dpcpp_nativecpu_##name(                        \
      type value, MakeGlobalType<__nativecpu_state> *s) {                      \
    s->field = value;                                                          \
  }                                                                            \
  static_assert(true)

// Subgroup setters
DefStateSetWithType(set_num_sub_groups, NumSubGroups, uint32_t);
DefStateSetWithType(set_sub_group_id, SubGroup_id, uint32_t);
DefStateSetWithType(set_max_sub_group_size, SubGroup_size, uint32_t);

#define DefineStateGetWithType(name, field, type)                              \
  DEVICE_EXTERNAL_C GET_PROPS USED type __dpcpp_nativecpu_##name(              \
      MakeGlobalType<const __nativecpu_state> *s) {                            \
    return s->field;                                                           \
  }                                                                            \
  static_assert(true)
#define DefineStateGet_U32(name, field)                                        \
  DefineStateGetWithType(name, field, uint32_t)

// Subgroup getters
DefineStateGet_U32(get_sub_group_id, SubGroup_id);
DefineStateGet_U32(get_sub_group_local_id, SubGroup_local_id);
DefineStateGet_U32(get_sub_group_size, SubGroup_size);
DefineStateGet_U32(get_max_sub_group_size, SubGroup_size);
DefineStateGet_U32(get_num_sub_groups, NumSubGroups);

#define DefineStateGetWithType2(name, field, rtype, ptype)                     \
  DEVICE_EXTERNAL_C GET_PROPS USED rtype __dpcpp_nativecpu_##name(             \
      ptype dim, MakeGlobalType<const __nativecpu_state> *s) {                 \
    return s->field[dim];                                                      \
  }                                                                            \
  static_assert(true)

#define DefineStateGet_U64(name, field)                                        \
  DefineStateGetWithType2(name, field, uint64_t, uint32_t)

// Workgroup getters
DefineStateGet_U64(get_global_id, MGlobal_id);
DefineStateGet_U64(get_global_range, MGlobal_range);
DefineStateGet_U64(get_global_offset, MGlobalOffset);
DefineStateGet_U64(get_local_id, MLocal_id);
DefineStateGet_U64(get_num_groups, MNumGroups);
DefineStateGet_U64(get_wg_size, MWorkGroup_size);
DefineStateGet_U64(get_wg_id, MWorkGroup_id);

DEVICE_EXTERNAL_C USED void
__dpcpp_nativecpu_set_local_id(uint32_t dim, uint64_t value,
                               MakeGlobalType<__nativecpu_state> *s) {
  s->MLocal_id[dim] = value;
  s->MGlobal_id[dim] = s->MWorkGroup_size[dim] * s->MWorkGroup_id[dim] +
                       s->MLocal_id[dim] + s->MGlobalOffset[dim];
}

#endif // __SYCL_NATIVE_CPU__
