
//==------ fallback_gsort.cpp - fallback implementation of group sort-------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "device.h"
#include "sort_helper.hpp"
#include <cstdint>
#include <functional>
#if defined(__SPIR__) || defined(__SPIRV__)

#define WG_JS_A(EP) __devicelib_default_work_group_joint_sort_ascending_##EP
#define WG_JS_D(EP) __devicelib_default_work_group_joint_sort_descending_##EP
#define WG_PS_CA(EP)                                                           \
  __devicelib_default_work_group_private_sort_close_ascending_##EP
#define WG_PS_CD(EP)                                                           \
  __devicelib_default_work_group_private_sort_close_descending_##EP
#define WG_PS_SA(EP)                                                           \
  __devicelib_default_work_group_private_sort_spread_ascending_##EP
#define WG_PS_SD(EP)                                                           \
  __devicelib_default_work_group_private_sort_spread_descending_##EP
#define SG_PS_A(EP) __devicelib_default_sub_group_private_sort_ascending_##EP
#define SG_PS_D(EP) __devicelib_default_sub_group_private_sort_descending_##EP

//============ default work grop joint sort for signed integer ===============
DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i8_u32_p1i8)(int8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i8_u32_p3i8)(int8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p3i8_u32_p1i8)(int8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p3i8_u32_p3i8)(int8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i8_u32_p1i8)(int8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i8_u32_p3i8)(int8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p3i8_u32_p1i8)(int8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p3i8_u32_p3i8)(int8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i16_u32_p1i8)(int16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i16_u32_p3i8)(int16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p3i16_u32_p1i8)(int16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p3i16_u32_p3i8)(int16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i16_u32_p1i8)(int16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i16_u32_p3i8)(int16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p3i16_u32_p1i8)(int16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p3i16_u32_p3i8)(int16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i32_u32_p1i8)(int32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i32_u32_p3i8)(int32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p3i32_u32_p1i8)(int32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p3i32_u32_p3i8)(int32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i32_u32_p1i8)(int32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i32_u32_p3i8)(int32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p3i32_u32_p1i8)(int32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p3i32_u32_p3i8)(int32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i64_u32_p1i8)(int64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i64_u32_p3i8)(int64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p3i64_u32_p1i8)(int64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p3i64_u32_p3i8)(int64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i64_u32_p1i8)(int64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i64_u32_p3i8)(int64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p3i64_u32_p1i8)(int64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p3i64_u32_p3i8)(int64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int64_t>{});
}

//=========== default work grop joint sort for unsigned integer ==============
DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u8_u32_p1i8)(uint8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u8_u32_p3i8)(uint8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p3u8_u32_p1i8)(uint8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p3u8_u32_p3i8)(uint8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u8_u32_p1i8)(uint8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u8_u32_p3i8)(uint8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p3u8_u32_p1i8)(uint8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p3u8_u32_p3i8)(uint8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u16_u32_p1i8)(uint16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u16_u32_p3i8)(uint16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p3u16_u32_p1i8)(uint16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p3u16_u32_p3i8)(uint16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u16_u32_p1i8)(uint16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u16_u32_p3i8)(uint16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p3u16_u32_p1i8)(uint16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p3u16_u32_p3i8)(uint16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u32_u32_p1i8)(uint32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u32_u32_p3i8)(uint32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p3u32_u32_p1i8)(uint32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p3u32_u32_p3i8)(uint32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u32_u32_p1i8)(uint32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u32_u32_p3i8)(uint32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p3u32_u32_p1i8)(uint32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p3u32_u32_p3i8)(uint32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u64_u32_p1i8)(uint64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u64_u32_p3i8)(uint64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p3u64_u32_p1i8)(uint64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p3u64_u32_p3i8)(uint64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u64_u32_p1i8)(uint64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u64_u32_p3i8)(uint64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p3u64_u32_p1i8)(uint64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p3u64_u32_p3i8)(uint64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint64_t>{});
}

//=============== default work grop joint sort for fp32 ======================
DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1f32_u32_p1i8)(float *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<float>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1f32_u32_p3i8)(float *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<float>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p3f32_u32_p1i8)(float *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<float>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p3f32_u32_p3i8)(float *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<float>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1f32_u32_p1i8)(float *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<float>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1f32_u32_p3i8)(float *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<float>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p3f32_u32_p1i8)(float *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<float>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p3f32_u32_p3i8)(float *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<float>{});
}

// TODO: split all f16 functions into separate libraries in case some platform
// doesn't support native fp16
//=============== default work grop joint sort for fp16 ======================
DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1f16_u32_p1i8)(_Float16 *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, [](_Float16 a, _Float16 b) { return (a < b); });
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1f16_u32_p3i8)(_Float16 *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, [](_Float16 a, _Float16 b) { return (a < b); });
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p3f16_u32_p1i8)(_Float16 *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, [](_Float16 a, _Float16 b) { return (a < b); });
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p3f16_u32_p3i8)(_Float16 *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, [](_Float16 a, _Float16 b) { return (a < b); });
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1f16_u32_p1i8)(_Float16 *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, [](_Float16 a, _Float16 b) { return (a > b); });
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1f16_u32_p3i8)(_Float16 *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, [](_Float16 a, _Float16 b) { return (a > b); });
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p3f16_u32_p1i8)(_Float16 *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, [](_Float16 a, _Float16 b) { return (a > b); });
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p3f16_u32_p3i8)(_Float16 *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, [](_Float16 a, _Float16 b) { return (a > b); });
}

//============ default work grop private sort for signed integer ==============
// Since 'first' should point to 'private' memory address space, it can only be
// decorated with 'p1'.
DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i8_u32_p1i8)(int8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i8_u32_p3i8)(int8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i8_u32_p1i8)(int8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i8_u32_p3i8)(int8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i8_u32_p1i8)(int8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i8_u32_p3i8)(int8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i8_u32_p1i8)(int8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i8_u32_p3i8)(int8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i16_u32_p1i8)(int16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i16_u32_p3i8)(int16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i16_u32_p1i8)(int16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i16_u32_p3i8)(int16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i16_u32_p1i8)(int16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i16_u32_p3i8)(int16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i16_u32_p1i8)(int16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i16_u32_p3i8)(int16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i32_u32_p1i8)(int32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i32_u32_p3i8)(int32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i32_u32_p1i8)(int32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i32_u32_p3i8)(int32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i32_u32_p1i8)(int32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i32_u32_p3i8)(int32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i32_u32_p1i8)(int32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i32_u32_p3i8)(int32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i64_u32_p1i8)(int64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i64_u32_p3i8)(int64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i64_u32_p1i8)(int64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i64_u32_p3i8)(int64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i64_u32_p1i8)(int64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i64_u32_p3i8)(int64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i64_u32_p1i8)(int64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i64_u32_p3i8)(int64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<int64_t>{});
}

//=========== default work grop private sort for unsigned integer =============

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u8_u32_p1i8)(uint8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u8_u32_p3i8)(uint8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u8_u32_p1i8)(uint8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u8_u32_p3i8)(uint8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u8_u32_p1i8)(uint8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u8_u32_p3i8)(uint8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u8_u32_p1i8)(uint8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u8_u32_p3i8)(uint8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u16_u32_p1i8)(uint16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u16_u32_p3i8)(uint16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u16_u32_p1i8)(uint16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u16_u32_p3i8)(uint16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u16_u32_p1i8)(uint16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u16_u32_p3i8)(uint16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u16_u32_p1i8)(uint16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u16_u32_p3i8)(uint16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u32_u32_p1i8)(uint32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u32_u32_p3i8)(uint32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u32_u32_p1i8)(uint32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u32_u32_p3i8)(uint32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u32_u32_p1i8)(uint32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u32_u32_p3i8)(uint32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u32_u32_p1i8)(uint32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u32_u32_p3i8)(uint32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u64_u32_p1i8)(uint64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u64_u32_p3i8)(uint64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u64_u32_p1i8)(uint64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u64_u32_p3i8)(uint64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u64_u32_p1i8)(uint64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u64_u32_p3i8)(uint64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u64_u32_p1i8)(uint64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u64_u32_p3i8)(uint64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<uint64_t>{});
}

//================= default work grop private sort for fp32 ====================

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1f32_u32_p1i8)(float *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<float>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1f32_u32_p3i8)(float *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<float>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1f32_u32_p1i8)(float *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<float>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1f32_u32_p3i8)(float *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<float>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1f32_u32_p1i8)(float *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<float>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1f32_u32_p3i8)(float *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<float>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1f32_u32_p1i8)(float *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<float>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1f32_u32_p3i8)(float *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<float>{});
}

//================= default work grop private sort for fp16 ====================

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1f16_u32_p1i8)(_Float16 *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch,
                           [](_Float16 a, _Float16 b) { return (a < b); });
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1f16_u32_p3i8)(_Float16 *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch,
                           [](_Float16 a, _Float16 b) { return (a < b); });
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1f16_u32_p1i8)(_Float16 *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch,
                           [](_Float16 a, _Float16 b) { return (a > b); });
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1f16_u32_p3i8)(_Float16 *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch,
                           [](_Float16 a, _Float16 b) { return (a > b); });
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1f16_u32_p1i8)(_Float16 *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch,
                            [](_Float16 a, _Float16 b) { return (a < b); });
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1f16_u32_p3i8)(_Float16 *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch,
                            [](_Float16 a, _Float16 b) { return (a < b); });
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1f16_u32_p1i8)(_Float16 *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch,
                            [](_Float16 a, _Float16 b) { return (a > b); });
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1f16_u32_p3i8)(_Float16 *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch,
                            [](_Float16 a, _Float16 b) { return (a > b); });
}

//============= default sub group private sort for signed integer =============
DEVICE_EXTERN_C_INLINE
int8_t SG_PS_A(i8_p1i8)(int8_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<int8_t>{});
}

int8_t SG_PS_A(i8_p3i8)(int8_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
int16_t SG_PS_A(i16_p1i8)(int16_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
int16_t SG_PS_A(i16_p3i8)(int16_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
int32_t SG_PS_A(i32_p1i8)(int32_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
int32_t SG_PS_A(i32_p3i8)(int32_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
int64_t SG_PS_A(i64_p1i8)(int64_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
int64_t SG_PS_A(i64_p3i8)(int64_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
uint8_t SG_PS_A(u8_p1i8)(uint8_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
uint8_t SG_PS_A(u8_p3i8)(uint8_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
uint16_t SG_PS_A(u16_p1i8)(uint16_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
uint16_t SG_PS_A(u16_p3i8)(uint16_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
uint32_t SG_PS_A(u32_p1i8)(uint32_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
uint32_t SG_PS_A(u32_p3i8)(uint32_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
uint64_t SG_PS_A(u64_p1i8)(uint64_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
uint64_t SG_PS_A(u64_p3i8)(uint64_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
float SG_PS_A(f32_p1i8)(float value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<float>{});
}

DEVICE_EXTERN_C_INLINE
float SG_PS_A(f32_p3i8)(float value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<float>{});
}

DEVICE_EXTERN_C_INLINE
_Float16 SG_PS_A(f16_p1i8)(_Float16 value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch,
                              [](_Float16 a, _Float16 b) { return (a < b); });
}

DEVICE_EXTERN_C_INLINE
_Float16 SG_PS_A(f16_p3i8)(_Float16 value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch,
                              [](_Float16 a, _Float16 b) { return (a < b); });
}

DEVICE_EXTERN_C_INLINE
int8_t SG_PS_D(i8_p1i8)(int8_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
int8_t SG_PS_D(i8_p3i8)(int8_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
int16_t SG_PS_D(i16_p1i8)(int16_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
int16_t SG_PS_D(i16_p3i8)(int16_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
int32_t SG_PS_D(i32_p1i8)(int32_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
int32_t SG_PS_D(i32_p3i8)(int32_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
int64_t SG_PS_D(i64_p1i8)(int64_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
int64_t SG_PS_D(i64_p3i8)(int64_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
uint8_t SG_PS_D(u8_p1i8)(uint8_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
uint8_t SG_PS_D(u8_p3i8)(uint8_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
uint16_t SG_PS_D(u16_p1i8)(uint16_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
uint16_t SG_PS_D(u16_p3i8)(uint16_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
uint32_t SG_PS_D(u32_p1i8)(uint32_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
uint32_t SG_PS_D(u32_p3i8)(uint32_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
uint64_t SG_PS_D(u64_p1i8)(uint64_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
uint64_t SG_PS_D(u64_p3i8)(uint64_t value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
float SG_PS_D(f32_p1i8)(float value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<float>{});
}

DEVICE_EXTERN_C_INLINE
float SG_PS_D(f32_p3i8)(float value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<float>{});
}

DEVICE_EXTERN_C_INLINE
_Float16 SG_PS_D(f16_p1i8)(_Float16 value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch,
                              [](_Float16 a, _Float16 b) { return (a > b); });
}

DEVICE_EXTERN_C_INLINE
_Float16 SG_PS_D(f16_p3i8)(_Float16 value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch,
                              [](_Float16 a, _Float16 b) { return (a > b); });
}

//========= default work grop joint sort for (uint32_t, uint32_t) ==============

// uint8_t as key type
DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u8_p1u8_u32_p1i8)(uint8_t *keys, uint8_t *vals, uint32_t n,
                                 uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u8_p1u8_u32_p1i8)(uint8_t *keys, uint8_t *vals, uint32_t n,
                                 uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint8_t>{});
}

// For int8_t values, the size and alignment are same as  uint8_t, we use same
// implementation as uint8_t values to reduce code size.
DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u8_p1i8_u32_p1i8)(uint8_t *keys, int8_t *vals, uint32_t n,
                                 uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u8_p1i8_u32_p1i8)(uint8_t *keys, int8_t *vals, uint32_t n,
                                 uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u8_p1u16_u32_p1i8)(uint8_t *keys, uint16_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u8_p1u16_u32_p1i8)(uint8_t *keys, uint16_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u8_p1i16_u32_p1i8)(uint8_t *keys, int16_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u8_p1i16_u32_p1i8)(uint8_t *keys, int16_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u8_p1u32_u32_p1i8)(uint8_t *keys, uint32_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u8_p1u32_u32_p1i8)(uint8_t *keys, uint32_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u8_p1i32_u32_p1i8)(uint8_t *keys, int32_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u8_p1i32_u32_p1i8)(uint8_t *keys, int32_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u8_p1u64_u32_p1i8)(uint8_t *keys, uint64_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u8_p1u64_u32_p1i8)(uint8_t *keys, uint64_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u8_p1i64_u32_p1i8)(uint8_t *keys, int64_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u8_p1i64_u32_p1i8)(uint8_t *keys, int64_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u8_p1f32_u32_p1i8)(uint8_t *keys, float *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u8_p1f32_u32_p1i8)(uint8_t *keys, float *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<uint8_t>{});
}

// int8_t as key
DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i8_p1u8_u32_p1i8)(int8_t *keys, uint8_t *vals, uint32_t n,
                                 uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i8_p1u8_u32_p1i8)(int8_t *keys, uint8_t *vals, uint32_t n,
                                 uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i8_p1i8_u32_p1i8)(int8_t *keys, int8_t *vals, uint32_t n,
                                 uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i8_p1i8_u32_p1i8)(int8_t *keys, int8_t *vals, uint32_t n,
                                 uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i8_p1u16_u32_p1i8)(int8_t *keys, uint16_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i8_p1u16_u32_p1i8)(int8_t *keys, uint16_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i8_p1i16_u32_p1i8)(int8_t *keys, int16_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i8_p1i16_u32_p1i8)(int8_t *keys, int16_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i8_p1u32_u32_p1i8)(int8_t *keys, uint32_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i8_p1u32_u32_p1i8)(int8_t *keys, uint32_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i8_p1i32_u32_p1i8)(int8_t *keys, int32_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i8_p1i32_u32_p1i8)(int8_t *keys, int32_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i8_p1u64_u32_p1i8)(int8_t *keys, uint64_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i8_p1u64_u32_p1i8)(int8_t *keys, uint64_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i8_p1i64_u32_p1i8)(int8_t *keys, int64_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i8_p1i64_u32_p1i8)(int8_t *keys, int64_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i8_p1f32_u32_p1i8)(int8_t *keys, float *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i8_p1f32_u32_p1i8)(int8_t *keys, float *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<int8_t>{});
}

// uint16_t as key type
DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u16_p1u8_u32_p1i8)(uint16_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u16_p1u8_u32_p1i8)(uint16_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u16_p1i8_u32_p1i8)(uint16_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u16_p1i8_u32_p1i8)(uint16_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u16_p1u16_u32_p1i8)(uint16_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u16_p1u16_u32_p1i8)(uint16_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u16_p1i16_u32_p1i8)(uint16_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u16_p1i16_u32_p1i8)(uint16_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u16_p1u32_u32_p1i8)(uint16_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u16_p1u32_u32_p1i8)(uint16_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u16_p1i32_u32_p1i8)(uint16_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u16_p1i32_u32_p1i8)(uint16_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u16_p1u64_u32_p1i8)(uint16_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u16_p1u64_u32_p1i8)(uint16_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u16_p1i64_u32_p1i8)(uint16_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u16_p1i64_u32_p1i8)(uint16_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u16_p1f32_u32_p1i8)(uint16_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u16_p1f32_u32_p1i8)(uint16_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<uint16_t>{});
}

// int16_t as key type
DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i16_p1u8_u32_p1i8)(int16_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i16_p1u8_u32_p1i8)(int16_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i16_p1i8_u32_p1i8)(int16_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i16_p1i8_u32_p1i8)(int16_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i16_p1u16_u32_p1i8)(int16_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i16_p1u16_u32_p1i8)(int16_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i16_p1i16_u32_p1i8)(int16_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i16_p1i16_u32_p1i8)(int16_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i16_p1u32_u32_p1i8)(int16_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i16_p1u32_u32_p1i8)(int16_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i16_p1i32_u32_p1i8)(int16_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i16_p1i32_u32_p1i8)(int16_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i16_p1u64_u32_p1i8)(int16_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i16_p1u64_u32_p1i8)(int16_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i16_p1i64_u32_p1i8)(int16_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i16_p1i64_u32_p1i8)(int16_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i16_p1f32_u32_p1i8)(int16_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i16_p1f32_u32_p1i8)(int16_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<int16_t>{});
}

// uint32_t as key type
DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u32_p1u8_u32_p1i8)(uint32_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u32_p1u8_u32_p1i8)(uint32_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u32_p1i8_u32_p1i8)(uint32_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u32_p1i8_u32_p1i8)(uint32_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u32_p1u16_u32_p1i8)(uint32_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u32_p1u16_u32_p1i8)(uint32_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u32_p1i16_u32_p1i8)(uint32_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u32_p1i16_u32_p1i8)(uint32_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u32_p1u32_u32_p1i8)(uint32_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u32_p1u32_u32_p1i8)(uint32_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u32_p1i32_u32_p1i8)(uint32_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u32_p1i32_u32_p1i8)(uint32_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u32_p1u64_u32_p1i8)(uint32_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u32_p1u64_u32_p1i8)(uint32_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u32_p1i64_u32_p1i8)(uint32_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u32_p1i64_u32_p1i8)(uint32_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u32_p1f32_u32_p1i8)(uint32_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u32_p1f32_u32_p1i8)(uint32_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<uint32_t>{});
}

// int32_t as key type
DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i32_p1u8_u32_p1i8)(int32_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i32_p1u8_u32_p1i8)(int32_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i32_p1i8_u32_p1i8)(int32_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i32_p1i8_u32_p1i8)(int32_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i32_p1u16_u32_p1i8)(int32_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i32_p1u16_u32_p1i8)(int32_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i32_p1i16_u32_p1i8)(int32_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i32_p1i16_u32_p1i8)(int32_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i32_p1u32_u32_p1i8)(int32_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i32_p1u32_u32_p1i8)(int32_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i32_p1i32_u32_p1i8)(int32_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i32_p1i32_u32_p1i8)(int32_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i32_p1u64_u32_p1i8)(int32_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i32_p1u64_u32_p1i8)(int32_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i32_p1i64_u32_p1i8)(int32_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i32_p1i64_u32_p1i8)(int32_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i32_p1f32_u32_p1i8)(int32_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i32_p1f32_u32_p1i8)(int32_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<int32_t>{});
}

// uint64_t as key type
DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u64_p1u8_u32_p1i8)(uint64_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u64_p1u8_u32_p1i8)(uint64_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u64_p1i8_u32_p1i8)(uint64_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u64_p1i8_u32_p1i8)(uint64_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u64_p1u16_u32_p1i8)(uint64_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u64_p1u16_u32_p1i8)(uint64_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u64_p1i16_u32_p1i8)(uint64_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u64_p1i16_u32_p1i8)(uint64_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u64_p1u32_u32_p1i8)(uint64_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u64_p1u32_u32_p1i8)(uint64_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u64_p1i32_u32_p1i8)(uint64_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u64_p1i32_u32_p1i8)(uint64_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u64_p1u64_u32_p1i8)(uint64_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u64_p1u64_u32_p1i8)(uint64_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u64_p1i64_u32_p1i8)(uint64_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u64_p1i64_u32_p1i8)(uint64_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u64_p1f32_u32_p1i8)(uint64_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u64_p1f32_u32_p1i8)(uint64_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<uint64_t>{});
}

// int64_t as key type
DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i64_p1u8_u32_p1i8)(int64_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i64_p1u8_u32_p1i8)(int64_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i64_p1i8_u32_p1i8)(int64_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i64_p1i8_u32_p1i8)(int64_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i64_p1u16_u32_p1i8)(int64_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i64_p1u16_u32_p1i8)(int64_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i64_p1i16_u32_p1i8)(int64_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i64_p1i16_u32_p1i8)(int64_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i64_p1u32_u32_p1i8)(int64_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i64_p1u32_u32_p1i8)(int64_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i64_p1i32_u32_p1i8)(int64_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i64_p1i32_u32_p1i8)(int64_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i64_p1u64_u32_p1i8)(int64_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i64_p1u64_u32_p1i8)(int64_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i64_p1i64_u32_p1i8)(int64_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i64_p1i64_u32_p1i8)(int64_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i64_p1f32_u32_p1i8)(int64_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i64_p1f32_u32_p1i8)(int64_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<int64_t>{});
}

// Work group private sorting algorithms.
DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u8_p1u8_u32_p1i8)(uint8_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u8_p1u8_u32_p1i8)(uint8_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u8_p1i8_u32_p1i8)(uint8_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u8_p1i8_u32_p1i8)(uint8_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u8_p1u16_u32_p1i8)(uint8_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u8_p1u16_u32_p1i8)(uint8_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u8_p1i16_u32_p1i8)(uint8_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u8_p1i16_u32_p1i8)(uint8_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u8_p1u32_u32_p1i8)(uint8_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u8_p1u32_u32_p1i8)(uint8_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u8_p1i32_u32_p1i8)(uint8_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u8_p1i32_u32_p1i8)(uint8_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u8_p1u64_u32_p1i8)(uint8_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u8_p1u64_u32_p1i8)(uint8_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u8_p1i64_u32_p1i8)(uint8_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u8_p1i64_u32_p1i8)(uint8_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u8_p1f32_u32_p1i8)(uint8_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u8_p1f32_u32_p1i8)(uint8_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i8_p1u8_u32_p1i8)(int8_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i8_p1u8_u32_p1i8)(int8_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i8_p1i8_u32_p1i8)(int8_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i8_p1i8_u32_p1i8)(int8_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i8_p1u16_u32_p1i8)(int8_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i8_p1u16_u32_p1i8)(int8_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i8_p1i16_u32_p1i8)(int8_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i8_p1i16_u32_p1i8)(int8_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i8_p1u32_u32_p1i8)(int8_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i8_p1u32_u32_p1i8)(int8_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i8_p1i32_u32_p1i8)(int8_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i8_p1i32_u32_p1i8)(int8_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i8_p1u64_u32_p1i8)(int8_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i8_p1u64_u32_p1i8)(int8_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i8_p1i64_u32_p1i8)(int8_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i8_p1i64_u32_p1i8)(int8_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i8_p1f32_u32_p1i8)(int8_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i8_p1f32_u32_p1i8)(int8_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u16_p1u8_u32_p1i8)(uint16_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u16_p1u8_u32_p1i8)(uint16_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u16_p1i8_u32_p1i8)(uint16_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u16_p1i8_u32_p1i8)(uint16_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u16_p1u16_u32_p1i8)(uint16_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u16_p1u16_u32_p1i8)(uint16_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u16_p1i16_u32_p1i8)(uint16_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u16_p1i16_u32_p1i8)(uint16_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u16_p1u32_u32_p1i8)(uint16_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u16_p1u32_u32_p1i8)(uint16_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u16_p1i32_u32_p1i8)(uint16_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u16_p1i32_u32_p1i8)(uint16_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u16_p1u64_u32_p1i8)(uint16_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u16_p1u64_u32_p1i8)(uint16_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u16_p1i64_u32_p1i8)(uint16_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u16_p1i64_u32_p1i8)(uint16_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u16_p1f32_u32_p1i8)(uint16_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u16_p1f32_u32_p1i8)(uint16_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i16_p1u8_u32_p1i8)(int16_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i16_p1u8_u32_p1i8)(int16_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i16_p1i8_u32_p1i8)(int16_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i16_p1i8_u32_p1i8)(int16_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i16_p1u16_u32_p1i8)(int16_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i16_p1u16_u32_p1i8)(int16_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i16_p1i16_u32_p1i8)(int16_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i16_p1i16_u32_p1i8)(int16_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i16_p1u32_u32_p1i8)(int16_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i16_p1u32_u32_p1i8)(int16_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i16_p1i32_u32_p1i8)(int16_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i16_p1i32_u32_p1i8)(int16_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i16_p1u64_u32_p1i8)(int16_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i16_p1u64_u32_p1i8)(int16_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i16_p1i64_u32_p1i8)(int16_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i16_p1i64_u32_p1i8)(int16_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i16_p1f32_u32_p1i8)(int16_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i16_p1f32_u32_p1i8)(int16_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u32_p1u8_u32_p1i8)(uint32_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u32_p1u8_u32_p1i8)(uint32_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u32_p1i8_u32_p1i8)(uint32_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u32_p1i8_u32_p1i8)(uint32_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u32_p1u16_u32_p1i8)(uint32_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u32_p1u16_u32_p1i8)(uint32_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u32_p1i16_u32_p1i8)(uint32_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u32_p1i16_u32_p1i8)(uint32_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u32_p1u32_u32_p1i8)(uint32_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u32_p1u32_u32_p1i8)(uint32_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u32_p1i32_u32_p1i8)(uint32_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u32_p1i32_u32_p1i8)(uint32_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u32_p1u64_u32_p1i8)(uint32_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u32_p1u64_u32_p1i8)(uint32_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u32_p1i64_u32_p1i8)(uint32_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u32_p1i64_u32_p1i8)(uint32_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u32_p1f32_u32_p1i8)(uint32_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u32_p1f32_u32_p1i8)(uint32_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i32_p1u8_u32_p1i8)(int32_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i32_p1u8_u32_p1i8)(int32_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i32_p1i8_u32_p1i8)(int32_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i32_p1i8_u32_p1i8)(int32_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i32_p1u16_u32_p1i8)(int32_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i32_p1u16_u32_p1i8)(int32_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i32_p1i16_u32_p1i8)(int32_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i32_p1i16_u32_p1i8)(int32_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i32_p1u32_u32_p1i8)(int32_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i32_p1u32_u32_p1i8)(int32_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i32_p1i32_u32_p1i8)(int32_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i32_p1i32_u32_p1i8)(int32_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i32_p1u64_u32_p1i8)(int32_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i32_p1u64_u32_p1i8)(int32_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i32_p1i64_u32_p1i8)(int32_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i32_p1i64_u32_p1i8)(int32_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i32_p1f32_u32_p1i8)(int32_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i32_p1f32_u32_p1i8)(int32_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u64_p1u8_u32_p1i8)(uint64_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u64_p1u8_u32_p1i8)(uint64_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u64_p1i8_u32_p1i8)(uint64_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u64_p1i8_u32_p1i8)(uint64_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u64_p1u16_u32_p1i8)(uint64_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u64_p1u16_u32_p1i8)(uint64_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u64_p1i16_u32_p1i8)(uint64_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u64_p1i16_u32_p1i8)(uint64_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u64_p1u32_u32_p1i8)(uint64_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u64_p1u32_u32_p1i8)(uint64_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u64_p1i32_u32_p1i8)(uint64_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u64_p1i32_u32_p1i8)(uint64_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u64_p1u64_u32_p1i8)(uint64_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u64_p1u64_u32_p1i8)(uint64_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u64_p1i64_u32_p1i8)(uint64_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u64_p1i64_u32_p1i8)(uint64_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u64_p1f32_u32_p1i8)(uint64_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u64_p1f32_u32_p1i8)(uint64_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i64_p1u8_u32_p1i8)(int64_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i64_p1u8_u32_p1i8)(int64_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i64_p1i8_u32_p1i8)(int64_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i64_p1i8_u32_p1i8)(int64_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i64_p1u16_u32_p1i8)(int64_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i64_p1u16_u32_p1i8)(int64_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i64_p1i16_u32_p1i8)(int64_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i64_p1i16_u32_p1i8)(int64_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i64_p1u32_u32_p1i8)(int64_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i64_p1u32_u32_p1i8)(int64_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i64_p1i32_u32_p1i8)(int64_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i64_p1i32_u32_p1i8)(int64_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i64_p1u64_u32_p1i8)(int64_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i64_p1u64_u32_p1i8)(int64_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i64_p1i64_u32_p1i8)(int64_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i64_p1i64_u32_p1i8)(int64_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i64_p1f32_u32_p1i8)(int64_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i64_p1f32_u32_p1i8)(int64_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u8_p1u8_u32_p1i8)(uint8_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u8_p1u8_u32_p1i8)(uint8_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u8_p1i8_u32_p1i8)(uint8_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u8_p1i8_u32_p1i8)(uint8_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u8_p1u16_u32_p1i8)(uint8_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u8_p1u16_u32_p1i8)(uint8_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u8_p1i16_u32_p1i8)(uint8_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u8_p1i16_u32_p1i8)(uint8_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u8_p1u32_u32_p1i8)(uint8_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u8_p1u32_u32_p1i8)(uint8_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u8_p1i32_u32_p1i8)(uint8_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u8_p1i32_u32_p1i8)(uint8_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u8_p1u64_u32_p1i8)(uint8_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u8_p1u64_u32_p1i8)(uint8_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u8_p1i64_u32_p1i8)(uint8_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u8_p1i64_u32_p1i8)(uint8_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u8_p1f32_u32_p1i8)(uint8_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u8_p1f32_u32_p1i8)(uint8_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i8_p1u8_u32_p1i8)(int8_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i8_p1u8_u32_p1i8)(int8_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i8_p1i8_u32_p1i8)(int8_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i8_p1i8_u32_p1i8)(int8_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i8_p1u16_u32_p1i8)(int8_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i8_p1u16_u32_p1i8)(int8_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i8_p1i16_u32_p1i8)(int8_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i8_p1i16_u32_p1i8)(int8_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i8_p1u32_u32_p1i8)(int8_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i8_p1u32_u32_p1i8)(int8_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i8_p1i32_u32_p1i8)(int8_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i8_p1i32_u32_p1i8)(int8_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i8_p1u64_u32_p1i8)(int8_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i8_p1u64_u32_p1i8)(int8_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i8_p1i64_u32_p1i8)(int8_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i8_p1i64_u32_p1i8)(int8_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i8_p1f32_u32_p1i8)(int8_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i8_p1f32_u32_p1i8)(int8_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u16_p1u8_u32_p1i8)(uint16_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u16_p1u8_u32_p1i8)(uint16_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u16_p1i8_u32_p1i8)(uint16_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u16_p1i8_u32_p1i8)(uint16_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u16_p1u16_u32_p1i8)(uint16_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u16_p1u16_u32_p1i8)(uint16_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u16_p1i16_u32_p1i8)(uint16_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u16_p1i16_u32_p1i8)(uint16_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u16_p1u32_u32_p1i8)(uint16_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u16_p1u32_u32_p1i8)(uint16_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u16_p1i32_u32_p1i8)(uint16_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u16_p1i32_u32_p1i8)(uint16_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u16_p1u64_u32_p1i8)(uint16_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u16_p1u64_u32_p1i8)(uint16_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u16_p1i64_u32_p1i8)(uint16_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u16_p1i64_u32_p1i8)(uint16_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u16_p1f32_u32_p1i8)(uint16_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u16_p1f32_u32_p1i8)(uint16_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i16_p1u8_u32_p1i8)(int16_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i16_p1u8_u32_p1i8)(int16_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i16_p1i8_u32_p1i8)(int16_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i16_p1i8_u32_p1i8)(int16_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i16_p1u16_u32_p1i8)(int16_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i16_p1u16_u32_p1i8)(int16_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i16_p1i16_u32_p1i8)(int16_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i16_p1i16_u32_p1i8)(int16_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i16_p1u32_u32_p1i8)(int16_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i16_p1u32_u32_p1i8)(int16_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i16_p1i32_u32_p1i8)(int16_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i16_p1i32_u32_p1i8)(int16_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i16_p1u64_u32_p1i8)(int16_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i16_p1u64_u32_p1i8)(int16_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i16_p1i64_u32_p1i8)(int16_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i16_p1i64_u32_p1i8)(int16_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i16_p1f32_u32_p1i8)(int16_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i16_p1f32_u32_p1i8)(int16_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u32_p1u8_u32_p1i8)(uint32_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u32_p1u8_u32_p1i8)(uint32_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u32_p1i8_u32_p1i8)(uint32_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u32_p1i8_u32_p1i8)(uint32_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u32_p1u16_u32_p1i8)(uint32_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u32_p1u16_u32_p1i8)(uint32_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u32_p1i16_u32_p1i8)(uint32_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u32_p1i16_u32_p1i8)(uint32_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u32_p1u32_u32_p1i8)(uint32_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u32_p1u32_u32_p1i8)(uint32_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u32_p1i32_u32_p1i8)(uint32_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u32_p1i32_u32_p1i8)(uint32_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u32_p1u64_u32_p1i8)(uint32_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u32_p1u64_u32_p1i8)(uint32_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u32_p1i64_u32_p1i8)(uint32_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u32_p1i64_u32_p1i8)(uint32_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u32_p1f32_u32_p1i8)(uint32_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u32_p1f32_u32_p1i8)(uint32_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i32_p1u8_u32_p1i8)(int32_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i32_p1u8_u32_p1i8)(int32_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i32_p1i8_u32_p1i8)(int32_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i32_p1i8_u32_p1i8)(int32_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i32_p1u16_u32_p1i8)(int32_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i32_p1u16_u32_p1i8)(int32_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i32_p1i16_u32_p1i8)(int32_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i32_p1i16_u32_p1i8)(int32_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i32_p1u32_u32_p1i8)(int32_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i32_p1u32_u32_p1i8)(int32_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i32_p1i32_u32_p1i8)(int32_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i32_p1i32_u32_p1i8)(int32_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i32_p1u64_u32_p1i8)(int32_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i32_p1u64_u32_p1i8)(int32_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i32_p1i64_u32_p1i8)(int32_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i32_p1i64_u32_p1i8)(int32_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i32_p1f32_u32_p1i8)(int32_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i32_p1f32_u32_p1i8)(int32_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u64_p1u8_u32_p1i8)(uint64_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u64_p1u8_u32_p1i8)(uint64_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u64_p1i8_u32_p1i8)(uint64_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u64_p1i8_u32_p1i8)(uint64_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u64_p1u16_u32_p1i8)(uint64_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u64_p1u16_u32_p1i8)(uint64_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u64_p1i16_u32_p1i8)(uint64_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u64_p1i16_u32_p1i8)(uint64_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u64_p1u32_u32_p1i8)(uint64_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u64_p1u32_u32_p1i8)(uint64_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u64_p1i32_u32_p1i8)(uint64_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u64_p1i32_u32_p1i8)(uint64_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u64_p1u64_u32_p1i8)(uint64_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u64_p1u64_u32_p1i8)(uint64_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u64_p1i64_u32_p1i8)(uint64_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u64_p1i64_u32_p1i8)(uint64_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u64_p1f32_u32_p1i8)(uint64_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u64_p1f32_u32_p1i8)(uint64_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i64_p1u8_u32_p1i8)(int64_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i64_p1u8_u32_p1i8)(int64_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i64_p1i8_u32_p1i8)(int64_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i64_p1i8_u32_p1i8)(int64_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i64_p1u16_u32_p1i8)(int64_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i64_p1u16_u32_p1i8)(int64_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i64_p1i16_u32_p1i8)(int64_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i64_p1i16_u32_p1i8)(int64_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i64_p1u32_u32_p1i8)(int64_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i64_p1u32_u32_p1i8)(int64_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i64_p1i32_u32_p1i8)(int64_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i64_p1i32_u32_p1i8)(int64_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i64_p1u64_u32_p1i8)(int64_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i64_p1u64_u32_p1i8)(int64_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i64_p1i64_u32_p1i8)(int64_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i64_p1i64_u32_p1i8)(int64_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i64_p1f32_u32_p1i8)(int64_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i64_p1f32_u32_p1i8)(int64_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u8_p1u8_u32_p3i8)(uint8_t *keys, uint8_t *vals, uint32_t n,
                                 uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u8_p1u8_u32_p3i8)(uint8_t *keys, uint8_t *vals, uint32_t n,
                                 uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u8_p1i8_u32_p3i8)(uint8_t *keys, int8_t *vals, uint32_t n,
                                 uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u8_p1i8_u32_p3i8)(uint8_t *keys, int8_t *vals, uint32_t n,
                                 uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u8_p1u16_u32_p3i8)(uint8_t *keys, uint16_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u8_p1u16_u32_p3i8)(uint8_t *keys, uint16_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u8_p1i16_u32_p3i8)(uint8_t *keys, int16_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u8_p1i16_u32_p3i8)(uint8_t *keys, int16_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u8_p1u32_u32_p3i8)(uint8_t *keys, uint32_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u8_p1u32_u32_p3i8)(uint8_t *keys, uint32_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u8_p1i32_u32_p3i8)(uint8_t *keys, int32_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u8_p1i32_u32_p3i8)(uint8_t *keys, int32_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u8_p1u64_u32_p3i8)(uint8_t *keys, uint64_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u8_p1u64_u32_p3i8)(uint8_t *keys, uint64_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u8_p1i64_u32_p3i8)(uint8_t *keys, int64_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u8_p1i64_u32_p3i8)(uint8_t *keys, int64_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u8_p1f32_u32_p3i8)(uint8_t *keys, float *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u8_p1f32_u32_p3i8)(uint8_t *keys, float *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<uint8_t>{});
}

// int8_t as key
DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i8_p1u8_u32_p3i8)(int8_t *keys, uint8_t *vals, uint32_t n,
                                 uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i8_p1u8_u32_p3i8)(int8_t *keys, uint8_t *vals, uint32_t n,
                                 uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i8_p1i8_u32_p3i8)(int8_t *keys, int8_t *vals, uint32_t n,
                                 uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i8_p1i8_u32_p3i8)(int8_t *keys, int8_t *vals, uint32_t n,
                                 uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i8_p1u16_u32_p3i8)(int8_t *keys, uint16_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i8_p1u16_u32_p3i8)(int8_t *keys, uint16_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i8_p1i16_u32_p3i8)(int8_t *keys, int16_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i8_p1i16_u32_p3i8)(int8_t *keys, int16_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i8_p1u32_u32_p3i8)(int8_t *keys, uint32_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i8_p1u32_u32_p3i8)(int8_t *keys, uint32_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i8_p1i32_u32_p3i8)(int8_t *keys, int32_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i8_p1i32_u32_p3i8)(int8_t *keys, int32_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i8_p1u64_u32_p3i8)(int8_t *keys, uint64_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i8_p1u64_u32_p3i8)(int8_t *keys, uint64_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i8_p1i64_u32_p3i8)(int8_t *keys, int64_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i8_p1i64_u32_p3i8)(int8_t *keys, int64_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i8_p1f32_u32_p3i8)(int8_t *keys, float *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i8_p1f32_u32_p3i8)(int8_t *keys, float *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<int8_t>{});
}

// uint16_t as key type
DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u16_p1u8_u32_p3i8)(uint16_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u16_p1u8_u32_p3i8)(uint16_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u16_p1i8_u32_p3i8)(uint16_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u16_p1i8_u32_p3i8)(uint16_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u16_p1u16_u32_p3i8)(uint16_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u16_p1u16_u32_p3i8)(uint16_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u16_p1i16_u32_p3i8)(uint16_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u16_p1i16_u32_p3i8)(uint16_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u16_p1u32_u32_p3i8)(uint16_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u16_p1u32_u32_p3i8)(uint16_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u16_p1i32_u32_p3i8)(uint16_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u16_p1i32_u32_p3i8)(uint16_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u16_p1u64_u32_p3i8)(uint16_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u16_p1u64_u32_p3i8)(uint16_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u16_p1i64_u32_p3i8)(uint16_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u16_p1i64_u32_p3i8)(uint16_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u16_p1f32_u32_p3i8)(uint16_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u16_p1f32_u32_p3i8)(uint16_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<uint16_t>{});
}

// int16_t as key type
DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i16_p1u8_u32_p3i8)(int16_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i16_p1u8_u32_p3i8)(int16_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i16_p1i8_u32_p3i8)(int16_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i16_p1i8_u32_p3i8)(int16_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i16_p1u16_u32_p3i8)(int16_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i16_p1u16_u32_p3i8)(int16_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i16_p1i16_u32_p3i8)(int16_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i16_p1i16_u32_p3i8)(int16_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i16_p1u32_u32_p3i8)(int16_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i16_p1u32_u32_p3i8)(int16_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i16_p1i32_u32_p3i8)(int16_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i16_p1i32_u32_p3i8)(int16_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i16_p1u64_u32_p3i8)(int16_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i16_p1u64_u32_p3i8)(int16_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i16_p1i64_u32_p3i8)(int16_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i16_p1i64_u32_p3i8)(int16_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i16_p1f32_u32_p3i8)(int16_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i16_p1f32_u32_p3i8)(int16_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<int16_t>{});
}

// uint32_t as key type
DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u32_p1u8_u32_p3i8)(uint32_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u32_p1u8_u32_p3i8)(uint32_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u32_p3i8_u32_p3i8)(uint32_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u32_p3i8_u32_p3i8)(uint32_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u32_p1u16_u32_p3i8)(uint32_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u32_p1u16_u32_p3i8)(uint32_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u32_p1i16_u32_p3i8)(uint32_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u32_p1i16_u32_p3i8)(uint32_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u32_p1u32_u32_p3i8)(uint32_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u32_p1u32_u32_p3i8)(uint32_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u32_p1i32_u32_p3i8)(uint32_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u32_p1i32_u32_p3i8)(uint32_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u32_p1u64_u32_p3i8)(uint32_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u32_p1u64_u32_p3i8)(uint32_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u32_p1i64_u32_p3i8)(uint32_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u32_p1i64_u32_p3i8)(uint32_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u32_p1f32_u32_p3i8)(uint32_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u32_p1f32_u32_p3i8)(uint32_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<uint32_t>{});
}

// int32_t as key type
DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i32_p1u8_u32_p3i8)(int32_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i32_p1u8_u32_p3i8)(int32_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i32_p1i8_u32_p3i8)(int32_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i32_p1i8_u32_p3i8)(int32_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i32_p1u16_u32_p3i8)(int32_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i32_p1u16_u32_p3i8)(int32_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i32_p1i16_u32_p3i8)(int32_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i32_p1i16_u32_p3i8)(int32_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i32_p1u32_u32_p3i8)(int32_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i32_p1u32_u32_p3i8)(int32_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i32_p1i32_u32_p3i8)(int32_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i32_p1i32_u32_p3i8)(int32_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i32_p1u64_u32_p3i8)(int32_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i32_p1u64_u32_p3i8)(int32_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i32_p1i64_u32_p3i8)(int32_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i32_p1i64_u32_p3i8)(int32_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i32_p1f32_u32_p3i8)(int32_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i32_p1f32_u32_p3i8)(int32_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<int32_t>{});
}

// uint64_t as key type
DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u64_p1u8_u32_p3i8)(uint64_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u64_p1u8_u32_p3i8)(uint64_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u64_p1i8_u32_p3i8)(uint64_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u64_p1i8_u32_p3i8)(uint64_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u64_p1u16_u32_p3i8)(uint64_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u64_p1u16_u32_p3i8)(uint64_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u64_p1i16_u32_p3i8)(uint64_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u64_p1i16_u32_p3i8)(uint64_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u64_p1u32_u32_p3i8)(uint64_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u64_p1u32_u32_p3i8)(uint64_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u64_p1i32_u32_p3i8)(uint64_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u64_p1i32_u32_p3i8)(uint64_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u64_p1u64_u32_p3i8)(uint64_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u64_p1u64_u32_p3i8)(uint64_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u64_p1i64_u32_p3i8)(uint64_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u64_p1i64_u32_p3i8)(uint64_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1u64_p1f32_u32_p3i8)(uint64_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1u64_p1f32_u32_p3i8)(uint64_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<uint64_t>{});
}

// int64_t as key type
DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i64_p1u8_u32_p3i8)(int64_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i64_p1u8_u32_p3i8)(int64_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i64_p1i8_u32_p3i8)(int64_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i64_p1i8_u32_p3i8)(int64_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint8_t *>(vals), n, scratch,
                       std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i64_p1u16_u32_p3i8)(int64_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i64_p1u16_u32_p3i8)(int64_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i64_p1i16_u32_p3i8)(int64_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i64_p1i16_u32_p3i8)(int64_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint16_t *>(vals), n, scratch,
                       std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i64_p1u32_u32_p3i8)(int64_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i64_p1u32_u32_p3i8)(int64_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i64_p1i32_u32_p3i8)(int64_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i64_p1i32_u32_p3i8)(int64_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i64_p1u64_u32_p3i8)(int64_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i64_p1u64_u32_p3i8)(int64_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, vals, n, scratch, std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i64_p1i64_u32_p3i8)(int64_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i64_p1i64_u32_p3i8)(int64_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint64_t *>(vals), n, scratch,
                       std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_A(p1i64_p1f32_u32_p3i8)(int64_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_JS_D(p1i64_p1f32_u32_p3i8)(int64_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  merge_sort_key_value(keys, reinterpret_cast<uint32_t *>(vals), n, scratch,
                       std::greater_equal<int64_t>{});
}

// Work group private sorting algorithms.
DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u8_p1u8_u32_p3i8)(uint8_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u8_p1u8_u32_p3i8)(uint8_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u8_p1i8_u32_p3i8)(uint8_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u8_p1i8_u32_p3i8)(uint8_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u8_p1u16_u32_p3i8)(uint8_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u8_p1u16_u32_p3i8)(uint8_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u8_p1i16_u32_p3i8)(uint8_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u8_p1i16_u32_p3i8)(uint8_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u8_p1u32_u32_p3i8)(uint8_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u8_p1u32_u32_p3i8)(uint8_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u8_p1i32_u32_p3i8)(uint8_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u8_p1i32_u32_p3i8)(uint8_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u8_p1u64_u32_p3i8)(uint8_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u8_p1u64_u32_p3i8)(uint8_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u8_p1i64_u32_p3i8)(uint8_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u8_p1i64_u32_p3i8)(uint8_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u8_p1f32_u32_p3i8)(uint8_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u8_p1f32_u32_p3i8)(uint8_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i8_p1u8_u32_p3i8)(int8_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i8_p1u8_u32_p3i8)(int8_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i8_p1i8_u32_p3i8)(int8_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i8_p1i8_u32_p3i8)(int8_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i8_p1u16_u32_p3i8)(int8_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i8_p1u16_u32_p3i8)(int8_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i8_p1i16_u32_p3i8)(int8_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i8_p1i16_u32_p3i8)(int8_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i8_p1u32_u32_p3i8)(int8_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i8_p1u32_u32_p3i8)(int8_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i8_p1i32_u32_p3i8)(int8_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i8_p1i32_u32_p3i8)(int8_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i8_p1u64_u32_p3i8)(int8_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i8_p1u64_u32_p3i8)(int8_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i8_p1i64_u32_p3i8)(int8_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i8_p1i64_u32_p3i8)(int8_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i8_p1f32_u32_p3i8)(int8_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i8_p1f32_u32_p3i8)(int8_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u16_p1u8_u32_p3i8)(uint16_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u16_p1u8_u32_p3i8)(uint16_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u16_p1i8_u32_p3i8)(uint16_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u16_p1i8_u32_p3i8)(uint16_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u16_p1u16_u32_p3i8)(uint16_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u16_p1u16_u32_p3i8)(uint16_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u16_p1i16_u32_p3i8)(uint16_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u16_p1i16_u32_p3i8)(uint16_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u16_p1u32_u32_p3i8)(uint16_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u16_p1u32_u32_p3i8)(uint16_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u16_p1i32_u32_p3i8)(uint16_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u16_p1i32_u32_p3i8)(uint16_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u16_p1u64_u32_p3i8)(uint16_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u16_p1u64_u32_p3i8)(uint16_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u16_p1i64_u32_p3i8)(uint16_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u16_p1i64_u32_p3i8)(uint16_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u16_p1f32_u32_p3i8)(uint16_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u16_p1f32_u32_p3i8)(uint16_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i16_p1u8_u32_p3i8)(int16_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i16_p1u8_u32_p3i8)(int16_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i16_p1i8_u32_p3i8)(int16_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i16_p1i8_u32_p3i8)(int16_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i16_p1u16_u32_p3i8)(int16_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i16_p1u16_u32_p3i8)(int16_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i16_p1i16_u32_p3i8)(int16_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i16_p1i16_u32_p3i8)(int16_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i16_p1u32_u32_p3i8)(int16_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i16_p1u32_u32_p3i8)(int16_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i16_p1i32_u32_p3i8)(int16_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i16_p1i32_u32_p3i8)(int16_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i16_p1u64_u32_p3i8)(int16_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i16_p1u64_u32_p3i8)(int16_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i16_p1i64_u32_p3i8)(int16_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i16_p1i64_u32_p3i8)(int16_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i16_p1f32_u32_p3i8)(int16_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i16_p1f32_u32_p3i8)(int16_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u32_p1u8_u32_p3i8)(uint32_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u32_p1u8_u32_p3i8)(uint32_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u32_p3i8_u32_p3i8)(uint32_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u32_p3i8_u32_p3i8)(uint32_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u32_p1u16_u32_p3i8)(uint32_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u32_p1u16_u32_p3i8)(uint32_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u32_p1i16_u32_p3i8)(uint32_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u32_p1i16_u32_p3i8)(uint32_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u32_p1u32_u32_p3i8)(uint32_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u32_p1u32_u32_p3i8)(uint32_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u32_p1i32_u32_p3i8)(uint32_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u32_p1i32_u32_p3i8)(uint32_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u32_p1u64_u32_p3i8)(uint32_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u32_p1u64_u32_p3i8)(uint32_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u32_p1i64_u32_p3i8)(uint32_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u32_p1i64_u32_p3i8)(uint32_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u32_p1f32_u32_p3i8)(uint32_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u32_p1f32_u32_p3i8)(uint32_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i32_p1u8_u32_p3i8)(int32_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i32_p1u8_u32_p3i8)(int32_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i32_p1i8_u32_p3i8)(int32_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i32_p1i8_u32_p3i8)(int32_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i32_p1u16_u32_p3i8)(int32_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i32_p1u16_u32_p3i8)(int32_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i32_p1i16_u32_p3i8)(int32_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i32_p1i16_u32_p3i8)(int32_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i32_p1u32_u32_p3i8)(int32_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i32_p1u32_u32_p3i8)(int32_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i32_p1i32_u32_p3i8)(int32_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i32_p1i32_u32_p3i8)(int32_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i32_p1u64_u32_p3i8)(int32_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i32_p1u64_u32_p3i8)(int32_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i32_p1i64_u32_p3i8)(int32_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i32_p1i64_u32_p3i8)(int32_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i32_p1f32_u32_p3i8)(int32_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i32_p1f32_u32_p3i8)(int32_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u64_p1u8_u32_p3i8)(uint64_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u64_p1u8_u32_p3i8)(uint64_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u64_p1i8_u32_p3i8)(uint64_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u64_p1i8_u32_p3i8)(uint64_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u64_p1u16_u32_p3i8)(uint64_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u64_p1u16_u32_p3i8)(uint64_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u64_p1i16_u32_p3i8)(uint64_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u64_p1i16_u32_p3i8)(uint64_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u64_p1u32_u32_p3i8)(uint64_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u64_p1u32_u32_p3i8)(uint64_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u64_p1i32_u32_p3i8)(uint64_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u64_p1i32_u32_p3i8)(uint64_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u64_p1u64_u32_p3i8)(uint64_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u64_p1u64_u32_p3i8)(uint64_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u64_p1i64_u32_p3i8)(uint64_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u64_p1i64_u32_p3i8)(uint64_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1u64_p1f32_u32_p3i8)(uint64_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1u64_p1f32_u32_p3i8)(uint64_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch,
                                     std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i64_p1u8_u32_p3i8)(int64_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i64_p1u8_u32_p3i8)(int64_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i64_p1i8_u32_p3i8)(int64_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i64_p1i8_u32_p3i8)(int64_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint8_t *>(vals), n,
                                     scratch, std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i64_p1u16_u32_p3i8)(int64_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i64_p1u16_u32_p3i8)(int64_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i64_p1i16_u32_p3i8)(int64_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i64_p1i16_u32_p3i8)(int64_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint16_t *>(vals),
                                     n, scratch, std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i64_p1u32_u32_p3i8)(int64_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i64_p1u32_u32_p3i8)(int64_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i64_p1i32_u32_p3i8)(int64_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i64_p1i32_u32_p3i8)(int64_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i64_p1u64_u32_p3i8)(int64_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i64_p1u64_u32_p3i8)(int64_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, vals, n, scratch,
                                     std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i64_p1i64_u32_p3i8)(int64_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i64_p1i64_u32_p3i8)(int64_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint64_t *>(vals),
                                     n, scratch, std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CA(p1i64_p1f32_u32_p3i8)(int64_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_CD(p1i64_p1f32_u32_p3i8)(int64_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_close(keys, reinterpret_cast<uint32_t *>(vals),
                                     n, scratch, std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u8_p1u8_u32_p3i8)(uint8_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u8_p1u8_u32_p3i8)(uint8_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u8_p1i8_u32_p3i8)(uint8_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u8_p1i8_u32_p3i8)(uint8_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u8_p1u16_u32_p3i8)(uint8_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u8_p1u16_u32_p3i8)(uint8_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u8_p1i16_u32_p3i8)(uint8_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u8_p1i16_u32_p3i8)(uint8_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u8_p1u32_u32_p3i8)(uint8_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u8_p1u32_u32_p3i8)(uint8_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u8_p1i32_u32_p3i8)(uint8_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u8_p1i32_u32_p3i8)(uint8_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u8_p1u64_u32_p3i8)(uint8_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u8_p1u64_u32_p3i8)(uint8_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u8_p1i64_u32_p3i8)(uint8_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u8_p1i64_u32_p3i8)(uint8_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u8_p1f32_u32_p3i8)(uint8_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u8_p1f32_u32_p3i8)(uint8_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i8_p1u8_u32_p3i8)(int8_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i8_p1u8_u32_p3i8)(int8_t *keys, uint8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i8_p1i8_u32_p3i8)(int8_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i8_p1i8_u32_p3i8)(int8_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i8_p1u16_u32_p3i8)(int8_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i8_p1u16_u32_p3i8)(int8_t *keys, uint16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i8_p1i16_u32_p3i8)(int8_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i8_p1i16_u32_p3i8)(int8_t *keys, int16_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i8_p1u32_u32_p3i8)(int8_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i8_p1u32_u32_p3i8)(int8_t *keys, uint32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i8_p1i32_u32_p3i8)(int8_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i8_p1i32_u32_p3i8)(int8_t *keys, int32_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i8_p1u64_u32_p3i8)(int8_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i8_p1u64_u32_p3i8)(int8_t *keys, uint64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i8_p1i64_u32_p3i8)(int8_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i8_p1i64_u32_p3i8)(int8_t *keys, int64_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i8_p1f32_u32_p3i8)(int8_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i8_p1f32_u32_p3i8)(int8_t *keys, float *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::greater_equal<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u16_p1u8_u32_p3i8)(uint16_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u16_p1u8_u32_p3i8)(uint16_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u16_p1i8_u32_p3i8)(uint16_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u16_p1i8_u32_p3i8)(uint16_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u16_p1u16_u32_p3i8)(uint16_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u16_p1u16_u32_p3i8)(uint16_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u16_p1i16_u32_p3i8)(uint16_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u16_p1i16_u32_p3i8)(uint16_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u16_p1u32_u32_p3i8)(uint16_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u16_p1u32_u32_p3i8)(uint16_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u16_p1i32_u32_p3i8)(uint16_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u16_p1i32_u32_p3i8)(uint16_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u16_p1u64_u32_p3i8)(uint16_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u16_p1u64_u32_p3i8)(uint16_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u16_p1i64_u32_p3i8)(uint16_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u16_p1i64_u32_p3i8)(uint16_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u16_p1f32_u32_p3i8)(uint16_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u16_p1f32_u32_p3i8)(uint16_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i16_p1u8_u32_p3i8)(int16_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i16_p1u8_u32_p3i8)(int16_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i16_p1i8_u32_p3i8)(int16_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i16_p1i8_u32_p3i8)(int16_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i16_p1u16_u32_p3i8)(int16_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i16_p1u16_u32_p3i8)(int16_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i16_p1i16_u32_p3i8)(int16_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i16_p1i16_u32_p3i8)(int16_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i16_p1u32_u32_p3i8)(int16_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i16_p1u32_u32_p3i8)(int16_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i16_p1i32_u32_p3i8)(int16_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i16_p1i32_u32_p3i8)(int16_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i16_p1u64_u32_p3i8)(int16_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i16_p1u64_u32_p3i8)(int16_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i16_p1i64_u32_p3i8)(int16_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i16_p1i64_u32_p3i8)(int16_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i16_p1f32_u32_p3i8)(int16_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i16_p1f32_u32_p3i8)(int16_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u32_p1u8_u32_p3i8)(uint32_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u32_p1u8_u32_p3i8)(uint32_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u32_p3i8_u32_p3i8)(uint32_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u32_p3i8_u32_p3i8)(uint32_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u32_p1u16_u32_p3i8)(uint32_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u32_p1u16_u32_p3i8)(uint32_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u32_p1i16_u32_p3i8)(uint32_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u32_p1i16_u32_p3i8)(uint32_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u32_p1u32_u32_p3i8)(uint32_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u32_p1u32_u32_p3i8)(uint32_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u32_p1i32_u32_p3i8)(uint32_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u32_p1i32_u32_p3i8)(uint32_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u32_p1u64_u32_p3i8)(uint32_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u32_p1u64_u32_p3i8)(uint32_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u32_p1i64_u32_p3i8)(uint32_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u32_p1i64_u32_p3i8)(uint32_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u32_p1f32_u32_p3i8)(uint32_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u32_p1f32_u32_p3i8)(uint32_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i32_p1u8_u32_p3i8)(int32_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i32_p1u8_u32_p3i8)(int32_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i32_p1i8_u32_p3i8)(int32_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i32_p1i8_u32_p3i8)(int32_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i32_p1u16_u32_p3i8)(int32_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i32_p1u16_u32_p3i8)(int32_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i32_p1i16_u32_p3i8)(int32_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i32_p1i16_u32_p3i8)(int32_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i32_p1u32_u32_p3i8)(int32_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i32_p1u32_u32_p3i8)(int32_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i32_p1i32_u32_p3i8)(int32_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i32_p1i32_u32_p3i8)(int32_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i32_p1u64_u32_p3i8)(int32_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i32_p1u64_u32_p3i8)(int32_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i32_p1i64_u32_p3i8)(int32_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i32_p1i64_u32_p3i8)(int32_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i32_p1f32_u32_p3i8)(int32_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i32_p1f32_u32_p3i8)(int32_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u64_p1u8_u32_p3i8)(uint64_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u64_p1u8_u32_p3i8)(uint64_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u64_p1i8_u32_p3i8)(uint64_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u64_p1i8_u32_p3i8)(uint64_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u64_p1u16_u32_p3i8)(uint64_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u64_p1u16_u32_p3i8)(uint64_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u64_p1i16_u32_p3i8)(uint64_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u64_p1i16_u32_p3i8)(uint64_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u64_p1u32_u32_p3i8)(uint64_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u64_p1u32_u32_p3i8)(uint64_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u64_p1i32_u32_p3i8)(uint64_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u64_p1i32_u32_p3i8)(uint64_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u64_p1u64_u32_p3i8)(uint64_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u64_p1u64_u32_p3i8)(uint64_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u64_p1i64_u32_p3i8)(uint64_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u64_p1i64_u32_p3i8)(uint64_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1u64_p1f32_u32_p3i8)(uint64_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1u64_p1f32_u32_p3i8)(uint64_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i64_p1u8_u32_p3i8)(int64_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i64_p1u8_u32_p3i8)(int64_t *keys, uint8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i64_p1i8_u32_p3i8)(int64_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i64_p1i8_u32_p3i8)(int64_t *keys, int8_t *vals, uint32_t n,
                                   uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint8_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i64_p1u16_u32_p3i8)(int64_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i64_p1u16_u32_p3i8)(int64_t *keys, uint16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i64_p1i16_u32_p3i8)(int64_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i64_p1i16_u32_p3i8)(int64_t *keys, int16_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint16_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i64_p1u32_u32_p3i8)(int64_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i64_p1u32_u32_p3i8)(int64_t *keys, uint32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i64_p1i32_u32_p3i8)(int64_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i64_p1i32_u32_p3i8)(int64_t *keys, int32_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i64_p1u64_u32_p3i8)(int64_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i64_p1u64_u32_p3i8)(int64_t *keys, uint64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, vals, n, scratch,
                                      std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i64_p1i64_u32_p3i8)(int64_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i64_p1i64_u32_p3i8)(int64_t *keys, int64_t *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint64_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SA(p1i64_p1f32_u32_p3i8)(int64_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch, std::less_equal<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void WG_PS_SD(p1i64_p1f32_u32_p3i8)(int64_t *keys, float *vals, uint32_t n,
                                    uint8_t *scratch) {
  private_merge_sort_key_value_spread(keys, reinterpret_cast<uint32_t *>(vals),
                                      n, scratch,
                                      std::greater_equal<int64_t>{});
}

#endif // __SPIR__ || __SPIRV__
