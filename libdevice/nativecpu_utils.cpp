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

#define OCL_LOCAL __attribute__((opencl_local))
#define OCL_GLOBAL __attribute__((opencl_global))

DEVICE_EXTERNAL OCL_LOCAL void *
__spirv_GenericCastToPtrExplicit_ToLocal(void *p, int) {
  return (OCL_LOCAL void *)p;
}

DEVICE_EXTERNAL OCL_GLOBAL void *
__spirv_GenericCastToPtrExplicit_ToGlobal(void *p, int) {
  return (OCL_GLOBAL void *)p;
}

DEVICE_EXTERN_C void __mux_work_group_barrier(uint32_t id, uint32_t scope,
                                              uint32_t semantics);
__SYCL_CONVERGENT__ DEVICE_EXTERNAL void
__spirv_ControlBarrier(uint32_t Execution, uint32_t Memory,
                       uint32_t Semantics) {
  if (__spv::Scope::Flag::Workgroup == Execution)
    // todo: check id and args; use mux constants
    __mux_work_group_barrier(0, Execution, Semantics);
}

DEVICE_EXTERN_C void __mux_mem_barrier(uint32_t scope, uint32_t semantics);
__SYCL_CONVERGENT__ DEVICE_EXTERNAL void
__spirv_MemoryBarrier(uint32_t Memory, uint32_t Semantics) {
  __mux_mem_barrier(Memory, Semantics);
}

// Turning clang format off here because it reorders macro invocations
// making the following code very difficult to read.
// clang-format off
#define DefSubgroupBlockINTEL1(Type, PType)                                    \
  template <>                                                                  \
  __SYCL_CONVERGENT__ DEVICE_EXTERNAL Type                                     \
  __spirv_SubgroupBlockReadINTEL<Type>(const OCL_GLOBAL PType *Ptr) noexcept { \
    return Ptr[__spirv_SubgroupLocalInvocationId()];                           \
  }                                                                            \
  template <>                                                                  \
  __SYCL_CONVERGENT__ DEVICE_EXTERNAL void                                     \
  __spirv_SubgroupBlockWriteINTEL<Type>(PType OCL_GLOBAL * ptr,                \
                                        Type v) noexcept {                     \
    ((Type*)ptr)[__spirv_SubgroupLocalInvocationId()]  = v;                    \
  }

#define DefSubgroupBlockINTEL_vt(Type, VT_name)                                \
  DefSubgroupBlockINTEL1(ncpu_types::vtypes<Type>::VT_name, Type)

#define DefSubgroupBlockINTEL(Type)                                            \
  DefSubgroupBlockINTEL1(Type, Type) DefSubgroupBlockINTEL_vt(Type, v2)        \
  DefSubgroupBlockINTEL_vt(Type, v4) DefSubgroupBlockINTEL_vt(Type, v8)

namespace ncpu_types {
template <class T> struct vtypes {
  using v2 = typename sycl::vec<T, 2>::vector_t;
  using v4 = typename sycl::vec<T, 4>::vector_t;
  using v8 = typename sycl::vec<T, 8>::vector_t;
};
} // namespace ncpu_types

DefSubgroupBlockINTEL(uint32_t) DefSubgroupBlockINTEL(uint64_t)
DefSubgroupBlockINTEL(uint8_t) DefSubgroupBlockINTEL(uint16_t)

#define DefineGOp1(spir_sfx, name)\
DEVICE_EXTERN_C bool __mux_sub_group_##name##_i1(bool);\
DEVICE_EXTERN_C bool __mux_work_group_##name##_i1(uint32_t id, bool val);\
DEVICE_EXTERNAL bool __spirv_Group ## spir_sfx(unsigned g, bool val) {\
  if (__spv::Scope::Flag::Subgroup == g)\
    return __mux_sub_group_##name##_i1(val);\
  else if (__spv::Scope::Flag::Workgroup == g)\
    return __mux_work_group_##name##_i1(0, val);\
  return false;\
}

DefineGOp1(Any, any)
DefineGOp1(All, all)


#define DefineGOp(Type, MuxType, spir_sfx, mux_sfx)                            \
  DEVICE_EXTERN_C MuxType __mux_sub_group_scan_inclusive_##mux_sfx(MuxType);   \
  DEVICE_EXTERN_C MuxType __mux_sub_group_scan_exclusive_##mux_sfx(MuxType);   \
  DEVICE_EXTERN_C MuxType __mux_sub_group_reduce_##mux_sfx(MuxType);           \
  DEVICE_EXTERN_C MuxType __mux_work_group_scan_exclusive_##mux_sfx(uint32_t,  \
                                                                    MuxType);  \
  DEVICE_EXTERN_C MuxType __mux_work_group_scan_inclusive_##mux_sfx(uint32_t,  \
                                                                    MuxType);  \
  DEVICE_EXTERN_C MuxType __mux_work_group_reduce_##mux_sfx(uint32_t, MuxType);\
  DEVICE_EXTERNAL Type __spirv_Group##spir_sfx(uint32_t g, uint32_t id,        \
                                               Type v) {                       \
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
  }

#define DefineSignedGOp(Name, MuxName, Bits)\
  DefineGOp(int##Bits##_t, int##Bits##_t, Name, MuxName##Bits)

#define DefineUnsignedGOp(Name, MuxName, Bits)\
  DefineGOp(uint##Bits##_t, int##Bits##_t, Name, MuxName##Bits)

#define Define_32_64(Define, Name, MuxName)                                    \
  Define(Name, MuxName, 32)                                                    \
  Define(Name, MuxName, 64)

// todo: add support for other integer and float types once there are tests
#define DefineIntGOps(Name, MuxName)                                           \
  Define_32_64(DefineSignedGOp,   Name, MuxName)                               \
  Define_32_64(DefineUnsignedGOp, Name, MuxName)

#define DefineFPGOps(Name, MuxName)                                            \
  DefineGOp(float, float, Name, MuxName##32)                                   \
  DefineGOp(_Float16 , _Float16 , Name, MuxName##16)                           \
  DefineGOp(double, double, Name, MuxName##64)

DefineIntGOps(IAdd,    add_i)
DefineIntGOps(IMulKHR, mul_i)

Define_32_64(DefineUnsignedGOp, UMin, umin_i)
Define_32_64(DefineUnsignedGOp, UMax, umax_i)
Define_32_64(DefineSignedGOp,   SMin, smin_i)
Define_32_64(DefineSignedGOp,   SMax, smax_i)

DefineFPGOps(FMulKHR, fmul_f)
DefineFPGOps(FAdd,    fadd_f)
DefineFPGOps(FMin,    fmin_f)
DefineFPGOps(FMax,    fmax_f)

#define DefineBitwiseGroupOp(Type, MuxType, mux_sfx)                          \
  DefineGOp(Type, MuxType, BitwiseOrKHR, or_##mux_sfx)                        \
  DefineGOp(Type, MuxType, BitwiseXorKHR, xor_##mux_sfx)                      \
  DefineGOp(Type, MuxType, BitwiseAndKHR, and_##mux_sfx)

DefineBitwiseGroupOp(int32_t, int32_t, i32)
DefineBitwiseGroupOp(uint32_t, int32_t, i32)
DefineBitwiseGroupOp(int64_t, int64_t, i64)
DefineBitwiseGroupOp(uint64_t, int64_t, i64)

#define DefineLogicalGroupOp(Type, MuxType, mux_sfx)                          \
  DefineGOp(Type, MuxType, LogicalOrKHR, logical_or_##mux_sfx)                \
  DefineGOp(Type, MuxType, LogicalXorKHR, logical_xor_##mux_sfx)              \
  DefineGOp(Type, MuxType, LogicalAndKHR, logical_and_##mux_sfx)

DefineLogicalGroupOp(bool, bool, i1)

#define DefineBroadcastMuxType(Type, Sfx, MuxType, IDType)                    \
  DEVICE_EXTERN_C MuxType __mux_work_group_broadcast_##Sfx(                   \
      int32_t id, MuxType val, uint64_t lidx, uint64_t lidy, uint64_t lidz);  \
  DEVICE_EXTERN_C MuxType __mux_sub_group_broadcast_##Sfx(MuxType val,        \
                                                          int32_t sg_lid);

#define DefineBroadCastImpl(Type, Sfx, MuxType, IDType)                       \
  DEVICE_EXTERNAL Type __spirv_GroupBroadcast(uint32_t g, Type v,             \
                                              IDType l) {                     \
    if (__spv::Scope::Flag::Subgroup == g)                                    \
      return __mux_sub_group_broadcast_##Sfx(v, l);                           \
    else                                                                      \
      return __mux_work_group_broadcast_##Sfx(0, v, l, 0, 0);                 \
  }                                                                           \
                                                                              \
  DEVICE_EXTERNAL Type __spirv_GroupBroadcast(uint32_t g, Type v,             \
                                         sycl::vec<IDType, 2>::vector_t l) {  \
    if (__spv::Scope::Flag::Subgroup == g)                                    \
      return __mux_sub_group_broadcast_##Sfx(v, l[0]);                        \
    else                                                                      \
      return __mux_work_group_broadcast_##Sfx(0, v, l[0], l[1], 0);           \
  }                                                                           \
                                                                              \
  DEVICE_EXTERNAL Type __spirv_GroupBroadcast(uint32_t g, Type v,             \
                                          sycl::vec<IDType, 3>::vector_t l) { \
    if (__spv::Scope::Flag::Subgroup == g)                                    \
      return __mux_sub_group_broadcast_##Sfx(v, l[0]);                        \
    else                                                                      \
      return __mux_work_group_broadcast_##Sfx(0, v, l[0], l[1], l[2]);        \
  }                                                                           \

#define DefineBroadCast(Type, Sfx, MuxType) \
   DefineBroadcastMuxType(Type, Sfx, MuxType, uint32_t) \
   DefineBroadcastMuxType(Type, Sfx, MuxType, uint64_t) \
   DefineBroadCastImpl(Type, Sfx, MuxType, uint32_t)    \
   DefineBroadCastImpl(Type, Sfx, MuxType, uint64_t)    \

DefineBroadCast(uint32_t, i32, int32_t)
DefineBroadCast(int32_t, i32, int32_t)
DefineBroadCast(float, f32, float)
DefineBroadCast(double, f64, double)
DefineBroadCast(uint64_t, i64, int64_t)
DefineBroadCast(int64_t, i64, int64_t)


#define DefShuffleINTEL(Type, Sfx, MuxType)                                    \
  DEVICE_EXTERN_C MuxType __mux_sub_group_shuffle_##Sfx(MuxType val,           \
                                                        int32_t lid);          \
  template <>                                                                  \
  DEVICE_EXTERNAL Type __spirv_SubgroupShuffleINTEL<Type>(                     \
      Type val, unsigned id) noexcept {                                        \
    return (Type)__mux_sub_group_shuffle_##Sfx((MuxType)val, id);              \
  }

#define DefShuffleUpINTEL(Type, Sfx, MuxType)                                  \
  DEVICE_EXTERN_C MuxType __mux_sub_group_shuffle_up_##Sfx(                    \
      MuxType prev, MuxType curr, int32_t delta);                              \
  template <>                                                                  \
  DEVICE_EXTERNAL Type __spirv_SubgroupShuffleUpINTEL<Type>(                   \
      Type prev, Type curr, unsigned delta) noexcept {                         \
    return (Type)__mux_sub_group_shuffle_up_##Sfx((MuxType)prev,               \
                                                  (MuxType)curr, delta);       \
  }

#define DefShuffleDownINTEL(Type, Sfx, MuxType)                                \
  DEVICE_EXTERN_C MuxType __mux_sub_group_shuffle_down_##Sfx(                  \
      MuxType curr, MuxType next, int32_t delta);                              \
  template <>                                                                  \
  DEVICE_EXTERNAL Type __spirv_SubgroupShuffleDownINTEL<Type>(                 \
      Type curr, Type next, unsigned delta) noexcept {                         \
    return (Type)__mux_sub_group_shuffle_down_##Sfx((MuxType)curr,             \
                                                    (MuxType)next, delta);     \
  }

#define DefShuffleXorINTEL(Type, Sfx, MuxType)                                 \
  DEVICE_EXTERN_C MuxType __mux_sub_group_shuffle_xor_##Sfx(MuxType val,       \
                                                            int32_t xor_val);  \
  template <>                                                                  \
  DEVICE_EXTERNAL Type __spirv_SubgroupShuffleXorINTEL<Type>(                  \
      Type data, unsigned value) noexcept {                                    \
    return (Type)__mux_sub_group_shuffle_xor_##Sfx((MuxType)data, value);      \
  }

#define DefShuffleINTEL_All(Type, Sfx, MuxType)                                \
  DefShuffleINTEL(Type, Sfx, MuxType)                                          \
  DefShuffleUpINTEL(Type, Sfx, MuxType)                                        \
  DefShuffleDownINTEL(Type, Sfx, MuxType)                                      \
  DefShuffleXorINTEL(Type, Sfx, MuxType)

DefShuffleINTEL_All(uint64_t, i64, int64_t)
DefShuffleINTEL_All(int64_t, i64, int64_t)
DefShuffleINTEL_All(int32_t, i32, int32_t)
DefShuffleINTEL_All(uint32_t, i32, int32_t)
DefShuffleINTEL_All(int16_t, i16, int16_t)
DefShuffleINTEL_All(uint16_t, i16, int16_t)
DefShuffleINTEL_All(int8_t, i8, int8_t)
DefShuffleINTEL_All(uint8_t, i8, int8_t)
DefShuffleINTEL_All(double, f64, double)
DefShuffleINTEL_All(float, f32, float)
DefShuffleINTEL_All(_Float16, f16, _Float16)

// Vector versions of shuffle are generated by the FixABIBuiltinsSYCLNativeCPU pass

#define Define2ArgForward(Type, Name, Callee)\
DEVICE_EXTERNAL Type Name(Type a, Type b) { return Callee(a,b);}

Define2ArgForward(uint64_t, __spirv_ocl_u_min, std::min)


#define GEN_u32(bname, muxname)                                                \
  DEVICE_EXTERN_C uint32_t muxname();                                          \
  DEVICE_EXTERNAL uint32_t bname() { return muxname(); }
// subgroup
GEN_u32(__spirv_SubgroupLocalInvocationId, __mux_get_sub_group_local_id)
GEN_u32(__spirv_SubgroupMaxSize, __mux_get_max_sub_group_size)
GEN_u32(__spirv_SubgroupId, __mux_get_sub_group_id)
GEN_u32(__spirv_NumSubgroups, __mux_get_num_sub_groups)
GEN_u32(__spirv_SubgroupSize, __mux_get_sub_group_size)

// I64_I32
#define GEN_p(bname, muxname, arg)                                             \
  DEVICE_EXTERN_C uint64_t muxname(uint32_t);                                  \
  DEVICE_EXTERNAL uint64_t bname() { return muxname(arg); }

#define GEN_xyz(bname, ncpu_name)                                              \
  GEN_p(bname##_x, ncpu_name, 0)                                               \
  GEN_p(bname##_y, ncpu_name, 1)                                               \
  GEN_p(bname##_z, ncpu_name, 2)

GEN_xyz(__spirv_GlobalInvocationId, __mux_get_global_id)
GEN_xyz(__spirv_GlobalSize, __mux_get_global_size)
GEN_xyz(__spirv_GlobalOffset, __mux_get_global_offset)
GEN_xyz(__spirv_LocalInvocationId, __mux_get_local_id)
GEN_xyz(__spirv_NumWorkgroups, __mux_get_num_groups)
GEN_xyz(__spirv_WorkgroupSize, __mux_get_local_size)
GEN_xyz(__spirv_WorkgroupId, __mux_get_group_id)

#define NCPUPREFIX(name) __dpcpp_nativecpu##name

template <class T> using MakeGlobalType =
  typename sycl::detail::DecoratedType < T, sycl::access::address_space::
                                                     global_space>::type;

#define DefStateSetWithType(name, field, type)                                 \
  DEVICE_EXTERNAL_C void NCPUPREFIX(name)(                                     \
      type value, MakeGlobalType<__nativecpu_state> *s) {                      \
    s->field = value;                                                          \
  }

// Subgroup setters
DefStateSetWithType(_set_num_sub_groups, NumSubGroups, uint32_t)
DefStateSetWithType(_set_sub_group_id, SubGroup_id, uint32_t)
DefStateSetWithType(_set_max_sub_group_size, SubGroup_size, uint32_t)

#define DefineStateGetWithType(name, field, type)\
  DEVICE_EXTERNAL_C type NCPUPREFIX(name)(                                     \
      MakeGlobalType<__nativecpu_state> *s) {                                  \
    return s->field;                                                           \
  }
#define DefineStateGet_U32(name, field)                                        \
  DefineStateGetWithType(name, field, uint32_t)

// Subgroup getters
DefineStateGet_U32(_get_sub_group_id, SubGroup_id)
DefineStateGet_U32(_get_sub_group_local_id, SubGroup_local_id)
DefineStateGet_U32(_get_sub_group_size, SubGroup_size)
DefineStateGet_U32(_get_max_sub_group_size, SubGroup_size)
DefineStateGet_U32(_get_num_sub_groups, NumSubGroups)

#define DefineStateGetWithType2(name, field, rtype, ptype)                     \
  DEVICE_EXTERNAL_C rtype NCPUPREFIX(name)(ptype dim,                          \
      MakeGlobalType<__nativecpu_state> *s) {                                  \
    return s->field[dim];                                                      \
  }

#define DefineStateGet_U64(name, field)                                        \
  DefineStateGetWithType2(name, field, uint64_t, uint32_t)

// Workgroup getters
DefineStateGet_U64(_get_global_id, MGlobal_id)
DefineStateGet_U64(_get_global_range, MGlobal_range)
DefineStateGet_U64(_get_global_offset, MGlobalOffset)
DefineStateGet_U64(_get_local_id, MLocal_id)
DefineStateGet_U64(_get_num_groups, MNumGroups)
DefineStateGet_U64(_get_wg_size, MWorkGroup_size)
DefineStateGet_U64(_get_wg_id, MWorkGroup_id)

DEVICE_EXTERNAL_C void
    __dpcpp_nativecpu_set_local_id(uint32_t dim, uint64_t value,
                                   MakeGlobalType<__nativecpu_state> *s) {
  s->MLocal_id[dim] = value;
  s->MGlobal_id[dim] = s->MWorkGroup_size[dim] * s->MWorkGroup_id[dim] +
                       s->MLocal_id[dim] + s->MGlobalOffset[dim];
}

#endif // __SYCL_NATIVE_CPU__
