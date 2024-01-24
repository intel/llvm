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
#define DEVICE_EXTERNAL SYCL_EXTERNAL __attribute__((always_inline))

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

DefineBroadCast(int, i32, int32_t) DefineBroadCast(unsigned, i32, int32_t)
    DefineBroadCast(float, f32, float)

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

        DefShuffleINTEL_All(uint64_t, i64, int64_t) DefShuffleINTEL_All(
            int64_t, i64, int64_t) DefShuffleINTEL_All(int32_t, i32, int32_t)
            DefShuffleINTEL_All(uint32_t, i32, int32_t)
                DefShuffleINTEL_All(int16_t, i16, int16_t)
                    DefShuffleINTEL_All(uint16_t, i16, int16_t)
                        DefShuffleINTEL_All(double, f64, double)
                            DefShuffleINTEL_All(float, f32, float)

#define DefineShuffleVec(T, N, Sfx, MuxType)                                   \
  using vt##T##N = sycl::detail::VecStorage<T, N>::DataType;                   \
  using vt##MuxType##N = sycl::detail::VecStorage<MuxType, N>::DataType;       \
  DefShuffleINTEL_All(vt##T##N, v##N##Sfx, vt##MuxType##N)

#define DefineShuffleVec2to16(Type, Sfx, MuxType)                              \
  DefineShuffleVec(Type, 2, Sfx, MuxType)                                      \
      DefineShuffleVec(Type, 4, Sfx, MuxType)                                  \
          DefineShuffleVec(Type, 8, Sfx, MuxType)                              \
              DefineShuffleVec(Type, 16, Sfx, MuxType)

                                DefineShuffleVec2to16(int, i32, int)
                                    DefineShuffleVec2to16(unsigned, i32, int)
                                        DefineShuffleVec2to16(float, f32, float)

#endif // __SYCL_NATIVE_CPU__
