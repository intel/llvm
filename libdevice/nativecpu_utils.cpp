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

#include "CL/__spirv/spirv_ops.hpp"
#include "device.h"
#include <cstdint>
#include <sycl/types.hpp>

struct __nativecpu_state {
  size_t MGlobal_id[3];
  size_t MGlobal_range[3];
  size_t MWorkGroup_size[3];
  size_t MWorkGroup_id[3];
  size_t MLocal_id[3];
  size_t MNumGroups[3];
  size_t MGlobalOffset[3];
  uint32_t NumSubGroups, SubGroup_id, SubGroup_local_id, SubGroup_size;
  __nativecpu_state(size_t globalR0, size_t globalR1, size_t globalR2,
                    size_t localR0, size_t localR1, size_t localR2,
                    size_t globalO0, size_t globalO1, size_t globalO2)
      : MGlobal_range{globalR0, globalR1, globalR2},
        MWorkGroup_size{localR0, localR1, localR2},
        MNumGroups{globalR0 / localR0, globalR1 / localR1, globalR2 / localR2},
        MGlobalOffset{globalO0, globalO1, globalO2} {
    MGlobal_id[0] = 0;
    MGlobal_id[1] = 0;
    MGlobal_id[2] = 0;
    MWorkGroup_id[0] = 0;
    MWorkGroup_id[1] = 0;
    MWorkGroup_id[2] = 0;
    MLocal_id[0] = 0;
    MLocal_id[1] = 0;
    MLocal_id[2] = 0;
    NumSubGroups = 32;
    SubGroup_id = 0;
    SubGroup_local_id = 0;
    SubGroup_size = 1;
  }

  void update(size_t group0, size_t group1, size_t group2, size_t local0,
              size_t local1, size_t local2) {
    MWorkGroup_id[0] = group0;
    MWorkGroup_id[1] = group1;
    MWorkGroup_id[2] = group2;
    MLocal_id[0] = local0;
    MLocal_id[1] = local1;
    MLocal_id[2] = local2;
    MGlobal_id[0] =
        MWorkGroup_size[0] * MWorkGroup_id[0] + MLocal_id[0] + MGlobalOffset[0];
    MGlobal_id[1] =
        MWorkGroup_size[1] * MWorkGroup_id[1] + MLocal_id[1] + MGlobalOffset[1];
    MGlobal_id[2] =
        MWorkGroup_size[2] * MWorkGroup_id[2] + MLocal_id[2] + MGlobalOffset[2];
  }

  void update(size_t group0, size_t group1, size_t group2) {
    MWorkGroup_id[0] = group0;
    MWorkGroup_id[1] = group1;
    MWorkGroup_id[2] = group2;
  }
};

#undef DEVICE_EXTERNAL
#undef DEVICE_EXTERN_C
#define DEVICE_EXTERN_C extern "C" SYCL_EXTERNAL
#define DEVICE_EXTERNAL_C DEVICE_EXTERN_C __attribute__((always_inline))
#define DEVICE_EXTERNAL SYCL_EXTERNAL __attribute__((always_inline))

#define OCL_LOCAL __attribute__((opencl_local))
#define OCL_GLOBAL __attribute__((opencl_global))

OCL_LOCAL void *__spirv_GenericCastToPtrExplicit_ToLocal(void *p, int) {
  return (OCL_LOCAL void *)p;
}

OCL_GLOBAL void *__spirv_GenericCastToPtrExplicit_ToGlobal(void *p, int) {
  return (OCL_GLOBAL void *)p;
}

#define DefSubgroupBlockINTEL1(Type, PType)                                    \
  template <>                                                                  \
  __SYCL_CONVERGENT__ DEVICE_EXTERNAL Type                                     \
  __spirv_SubgroupBlockReadINTEL<Type>(const OCL_GLOBAL PType *Ptr) noexcept { \
    return *Ptr;                                                               \
  }                                                                            \
  template <>                                                                  \
  __SYCL_CONVERGENT__ DEVICE_EXTERNAL void                                     \
  __spirv_SubgroupBlockWriteINTEL<Type>(PType OCL_GLOBAL * ptr,                \
                                        Type v) noexcept {                     \
    *(Type *)ptr = v;                                                          \
  }

#define DefSubgroupBlockINTEL_vt(Type, VT_name)                                \
  DefSubgroupBlockINTEL1(ncpu_types::vtypes<Type>::VT_name, Type)

#define DefSubgroupBlockINTEL(Type)                                            \
  DefSubgroupBlockINTEL1(Type, Type) DefSubgroupBlockINTEL_vt(Type, v2)        \
      DefSubgroupBlockINTEL_vt(Type, v4) DefSubgroupBlockINTEL_vt(Type, v8)

namespace ncpu_types {
template <class T> struct vtypes {
  using v2 = __ocl_vec_t<T, 2>;
  using v4 = __ocl_vec_t<T, 4>;
  using v8 = __ocl_vec_t<T, 8>;
};
} // namespace ncpu_types

DefSubgroupBlockINTEL(unsigned) DefSubgroupBlockINTEL(unsigned __int64)
    DefSubgroupBlockINTEL(unsigned char) DefSubgroupBlockINTEL(unsigned short)

    // i1 @__mux_sub_group_any_i1(i1 %x)
    // i1 @__mux_work_group_any_i1(i32 %id, i1 %x)
    // i1 @__mux_vec_group_any_v4i1(<4 x i1> %x)

    DEVICE_EXTERN_C bool __mux_work_group_any_i1(unsigned, bool);
DEVICE_EXTERNAL bool __spirv_GroupAny(unsigned id, bool val) {
  return __mux_work_group_any_i1(id, val);
}

DEVICE_EXTERN_C bool __mux_work_group_all_i1(unsigned, bool);
DEVICE_EXTERNAL bool __spirv_GroupAll(unsigned id, bool val) {
  return __mux_work_group_all_i1(id, val);
}

#define DefineScan(Type, MuxType, spir_sfx, mux_sfx)                           \
  DEVICE_EXTERN_C MuxType __mux_sub_group_scan_inclusive_##mux_sfx(            \
      /*int,*/ MuxType);                                                       \
  DEVICE_EXTERNAL Type __spirv_Group##spir_sfx(unsigned g, unsigned id,        \
                                               Type v) {                       \
    if (__spv::Scope::Flag::Subgroup == g)                                     \
      return __mux_sub_group_scan_inclusive_##mux_sfx(/*id,*/ v);              \
    return Type(); /*TODO*/                                                    \
  }

DefineScan(int, int32_t, IAdd,
           add_i32) DefineScan(unsigned, int32_t, IAdd,
                               add_i32) DefineScan(float, float, FAdd, fadd_f32)

#define DefineReduce(Type, MuxType, spir_sfx, mux_sfx)                         \
  DEVICE_EXTERN_C MuxType __mux_sub_group_reduce_##mux_sfx(MuxType);           \
  DEVICE_EXTERNAL Type __spirv_Group##spir_sfx(unsigned g, unsigned id,        \
                                               Type v) {                       \
    if (__spv::Scope::Flag::Subgroup == g)                                     \
      return __mux_sub_group_reduce_##mux_sfx(v);                              \
    return Type(); /*TODO*/                                                    \
  }

    DefineReduce(double, double, FMulKHR, fmul_f64) DefineReduce(
        double, double, FAdd,
        fadd_f64) DefineReduce(double, double, FMin,
                               fmin_f64) DefineReduce(double, double, FMax,
                                                      fmax_f64)

        DefineReduce(int, int, IMulKHR,
                     mul_i32) DefineReduce(unsigned, int, IMulKHR,
                                           mul_i32) DefineReduce(float, float,
                                                                 FMulKHR,
                                                                 fmul_f32)

            DefineReduce(int, int, SMin, smin_i32) DefineReduce(int, int, SMax,
                                                                smax_i32)
                DefineReduce(unsigned, int, UMin, umin_i32) DefineReduce(
                    unsigned, int, UMax,
                    umax_i32) DefineReduce(float, float, FMin,
                                           fmin_f32) DefineReduce(float, float,
                                                                  FMax,
                                                                  fmax_f32)

#define DefineBitwiseReduce(Type, MuxType, mux_sfx)                            \
  DefineReduce(Type, MuxType, BitwiseOrKHR, or_##mux_sfx)                      \
      DefineReduce(Type, MuxType, BitwiseXorKHR, xor_##mux_sfx)                \
          DefineReduce(Type, MuxType, BitwiseAndKHR, and_##mux_sfx)

                    DefineBitwiseReduce(int, int, i32) DefineBitwiseReduce(
                        unsigned, int,
                        i32) DefineBitwiseReduce(int64_t, int64_t,
                                                 i64) DefineBitwiseReduce(uint64_t,
                                                                          int64_t,
                                                                          i64)

#define DefineBroadCast(Type, Sfx, MuxType)                                    \
  DEVICE_EXTERN_C MuxType __mux_work_group_broadcast_##Sfx(                    \
      int32_t id, MuxType val, int64_t lidx, int64_t lidy, int64_t lidz);      \
  DEVICE_EXTERN_C MuxType __mux_sub_group_broadcast_##Sfx(MuxType val,         \
                                                          int32_t sg_lid);     \
  DEVICE_EXTERNAL Type __spirv_GroupBroadcast(unsigned g, Type v,              \
                                              unsigned l) {                    \
    if (__spv::Scope::Flag::Subgroup == g)                                     \
      return __mux_sub_group_broadcast_##Sfx(v, l);                            \
    return Type(); /*TODO*/                                                    \
  }

                        DefineBroadCast(int, i32, int32_t) DefineBroadCast(
                            unsigned, i32, int32_t) DefineBroadCast(float, f32,
                                                                    float)
                            DefineBroadCast(double, f64, double)

// defining subgroup builtins

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
  DefShuffleINTEL(Type, Sfx, MuxType) DefShuffleUpINTEL(Type, Sfx, MuxType)    \
      DefShuffleDownINTEL(Type, Sfx, MuxType)                                  \
          DefShuffleXorINTEL(Type, Sfx, MuxType)

                                DefShuffleINTEL_All(
                                    uint64_t, i64,
                                    int64_t) DefShuffleINTEL_All(int64_t, i64,
                                                                 int64_t)
                                    DefShuffleINTEL_All(
                                        int32_t, i32,
                                        int32_t) DefShuffleINTEL_All(uint32_t,
                                                                     i32,
                                                                     int32_t)
                                        DefShuffleINTEL_All(
                                            int16_t, i16,
                                            int16_t) DefShuffleINTEL_All(uint16_t,
                                                                         i16,
                                                                         int16_t)
                                            DefShuffleINTEL_All(double, f64,
                                                                double)
                                                DefShuffleINTEL_All(float, f32,
                                                                    float)

#define DefineShuffleVec(T, N, Sfx, MuxType)                                   \
  using vt##T##N = sycl::detail::VecStorage<T, N>::DataType;                   \
  using vt##MuxType##N = sycl::detail::VecStorage<MuxType, N>::DataType;       \
  DefShuffleINTEL_All(vt##T##N, v##N##Sfx, vt##MuxType##N)

#define DefineShuffleVec2to16(Type, Sfx, MuxType)                              \
  DefineShuffleVec(Type, 2, Sfx, MuxType)                                      \
      DefineShuffleVec(Type, 4, Sfx, MuxType)                                  \
          DefineShuffleVec(Type, 8, Sfx, MuxType)                              \
              DefineShuffleVec(Type, 16, Sfx, MuxType)

                                                    DefineShuffleVec2to16(int,
                                                                          i32,
                                                                          int)
                                                        DefineShuffleVec2to16(
                                                            unsigned, i32, int)
                                                            DefineShuffleVec2to16(
                                                                float, f32,
                                                                float)

#define GEN_u32(bname, muxname)                                                \
  DEVICE_EXTERN_C uint32_t muxname();                                          \
  DEVICE_EXTERNAL uint32_t bname() { return muxname(); }
    // subgroup
    GEN_u32(__spirv_SubgroupLocalInvocationId,
            __mux_get_sub_group_local_id) GEN_u32(__spirv_SubgroupMaxSize,
                                                  __mux_get_max_sub_group_size)
        GEN_u32(__spirv_SubgroupId, __mux_get_sub_group_id) GEN_u32(
            __spirv_NumSubgroups,
            __mux_get_num_sub_groups) GEN_u32(__spirv_SubgroupSize,
                                              __mux_get_sub_group_size)

// I64_I32
#define GEN_p(bname, muxname, arg)                                             \
  DEVICE_EXTERN_C uint64_t muxname(uint32_t);                                  \
  DEVICE_EXTERNAL uint64_t bname() { return muxname(arg); }

#define GEN_xyz(bname, ncpu_name)                                              \
  GEN_p(bname##_x, ncpu_name, 0) GEN_p(bname##_y, ncpu_name, 1)                \
      GEN_p(bname##_z, ncpu_name, 2)

            GEN_xyz(__spirv_GlobalInvocationId,
                    __mux_get_global_id) GEN_xyz(__spirv_GlobalSize,
                                                 __mux_get_global_size)
                GEN_xyz(__spirv_GlobalOffset, __mux_get_global_offset) GEN_xyz(
                    __spirv_LocalInvocationId,
                    __mux_get_local_id) GEN_xyz(__spirv_NumWorkgroups,
                                                __mux_get_num_groups)
                    GEN_xyz(__spirv_WorkgroupSize,
                            __mux_get_local_size) GEN_xyz(__spirv_WorkgroupId,
                                                          __mux_get_group_id)

#define DefStateSetWithType(name, field, type)                                 \
  DEVICE_EXTERNAL_C void name(type value, __nativecpu_state *s) {              \
    s->field = value;                                                          \
  }

                        DefStateSetWithType(
                            __dpcpp_nativecpu_set_num_sub_groups, NumSubGroups,
                            uint32_t)
                            DefStateSetWithType(
                                __dpcpp_nativecpu_set_sub_group_id, SubGroup_id,
                                uint32_t)
                                DefStateSetWithType(
                                    __dpcpp_nativecpu_set_max_sub_group_size,
                                    SubGroup_size, uint32_t)

#endif // __SYCL_NATIVE_CPU__
