
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
#if defined(__SPIR__)

//============ default work grop joint sort for signed integer ===============
DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p1i8_u32_p1i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p1i8_u32_p3i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3i8_u32_p1i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3i8_u32_p3i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p1i8_u32_p1i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p1i8_u32_p3i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p3i8_u32_p1i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p3i8_u32_p3i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p1i16_u32_p1i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p1i16_u32_p3i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3i16_u32_p1i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3i16_u32_p3i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p1i16_u32_p1i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p1i16_u32_p3i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p3i16_u32_p1i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p3i16_u32_p3i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p1i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p1i32_u32_p3i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3i32_u32_p3i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p1i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p1i32_u32_p3i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p3i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p3i32_u32_p3i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int32_t>{});
}

void __devicelib_default_work_group_joint_sort_ascending_p1i64_u32_p1i8(
    int64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p1i64_u32_p3i8(
    int64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3i64_u32_p1i8(
    int64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3i64_u32_p3i8(
    int64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p1i64_u32_p1i8(
    int64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p1i64_u32_p3i8(
    int64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p3i64_u32_p1i8(
    int64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p3i64_u32_p3i8(
    int64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<int64_t>{});
}

//=========== default work grop joint sort for unsigned integer ==============
DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p1u8_u32_p1i8(
    uint8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p1u8_u32_p3i8(
    uint8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3u8_u32_p1i8(
    uint8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3u8_u32_p3i8(
    uint8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p1u8_u32_p1i8(
    uint8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p1u8_u32_p3i8(
    uint8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p3u8_u32_p1i8(
    uint8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p3u8_u32_p3i8(
    uint8_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p1u16_u32_p1i8(
    uint16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p1u16_u32_p3i8(
    uint16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3u16_u32_p1i8(
    uint16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3u16_u32_p3i8(
    uint16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p1u16_u32_p1i8(
    uint16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p1u16_u32_p3i8(
    uint16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p3u16_u32_p1i8(
    uint16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p3u16_u32_p3i8(
    uint16_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p1u32_u32_p1i8(
    uint32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p1u32_u32_p3i8(
    uint32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3u32_u32_p1i8(
    uint32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3u32_u32_p3i8(
    uint32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p1u32_u32_p1i8(
    uint32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p1u32_u32_p3i8(
    uint32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p3u32_u32_p1i8(
    uint32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p3u32_u32_p3i8(
    uint32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint32_t>{});
}

void __devicelib_default_work_group_joint_sort_ascending_p1u64_u32_p1i8(
    uint64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p1u64_u32_p3i8(
    uint64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3u64_u32_p1i8(
    uint64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3u64_u32_p3i8(
    uint64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p1u64_u32_p1i8(
    uint64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p1u64_u32_p3i8(
    uint64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p3u64_u32_p1i8(
    uint64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p3u64_u32_p3i8(
    uint64_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<uint64_t>{});
}

//=============== default work grop joint sort for fp32 ======================
DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p1f32_u32_p1i8(
    float *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<float>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p1f32_u32_p3i8(
    float *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<float>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3f32_u32_p1i8(
    float *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<float>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3f32_u32_p3i8(
    float *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::less<float>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p1f32_u32_p1i8(
    float *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<float>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p1f32_u32_p3i8(
    float *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<float>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p3f32_u32_p1i8(
    float *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<float>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p3f32_u32_p3i8(
    float *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, std::greater<float>{});
}

// TODO: split all f16 functions into separate libraries in case some platform
// doesn't support native fp16
//=============== default work grop joint sort for fp16 ======================
DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p1f16_u32_p1i8(
    _Float16 *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, [](_Float16 a, _Float16 b) { return (a < b); });
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p1f16_u32_p3i8(
    _Float16 *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, [](_Float16 a, _Float16 b) { return (a < b); });
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3f16_u32_p1i8(
    _Float16 *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, [](_Float16 a, _Float16 b) { return (a < b); });
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3f16_u32_p3i8(
    _Float16 *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, [](_Float16 a, _Float16 b) { return (a < b); });
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p1f16_u32_p1i8(
    _Float16 *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, [](_Float16 a, _Float16 b) { return (a > b); });
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p1f16_u32_p3i8(
    _Float16 *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, [](_Float16 a, _Float16 b) { return (a > b); });
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p3f16_u32_p1i8(
    _Float16 *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, [](_Float16 a, _Float16 b) { return (a > b); });
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_descending_p3f16_u32_p3i8(
    _Float16 *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch, [](_Float16 a, _Float16 b) { return (a > b); });
}

//============ default work grop private sort for signed integer ==============
// Since 'first' should point to 'private' memory address space, it can only be
// decorated with 'p1'.
DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_ascending_p1i8_u32_p1i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_ascending_p1i8_u32_p3i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_descending_p1i8_u32_p1i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_descending_p1i8_u32_p3i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_ascending_p1i8_u32_p1i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_ascending_p1i8_u32_p3i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_descending_p1i8_u32_p1i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_descending_p1i8_u32_p3i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_ascending_p1i16_u32_p1i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_ascending_p1i16_u32_p3i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_descending_p1i16_u32_p1i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_descending_p1i16_u32_p3i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_ascending_p1i16_u32_p1i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_ascending_p1i16_u32_p3i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_descending_p1i16_u32_p1i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_descending_p1i16_u32_p3i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_ascending_p1i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_ascending_p1i32_u32_p3i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_descending_p1i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_descending_p1i32_u32_p3i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_ascending_p1i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_ascending_p1i32_u32_p3i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_descending_p1i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_descending_p1i32_u32_p3i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_ascending_p1i64_u32_p1i8(
    int64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_ascending_p1i64_u32_p3i8(
    int64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_descending_p1i64_u32_p1i8(
    int64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_descending_p1i64_u32_p3i8(
    int64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_ascending_p1i64_u32_p1i8(
    int64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_ascending_p1i64_u32_p3i8(
    int64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_descending_p1i64_u32_p1i8(
    int64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_descending_p1i64_u32_p3i8(
    int64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<int64_t>{});
}

//=========== default work grop private sort for unsigned integer =============

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_ascending_p1u8_u32_p1i8(
    uint8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_ascending_p1u8_u32_p3i8(
    uint8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_descending_p1u8_u32_p1i8(
    uint8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_descending_p1u8_u32_p3i8(
    uint8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_ascending_p1u8_u32_p1i8(
    uint8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_ascending_p1u8_u32_p3i8(
    uint8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_descending_p1u8_u32_p1i8(
    uint8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_descending_p1u8_u32_p3i8(
    uint8_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_ascending_p1u16_u32_p1i8(
    uint16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_ascending_p1u16_u32_p3i8(
    uint16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_descending_p1u16_u32_p1i8(
    uint16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_descending_p1u16_u32_p3i8(
    uint16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_ascending_p1u16_u32_p1i8(
    uint16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_ascending_p1u16_u32_p3i8(
    uint16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_descending_p1u16_u32_p1i8(
    uint16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_descending_p1u16_u32_p3i8(
    uint16_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_ascending_p1u32_u32_p1i8(
    uint32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_ascending_p1u32_u32_p3i8(
    uint32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_descending_p1u32_u32_p1i8(
    uint32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_descending_p1u32_u32_p3i8(
    uint32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_ascending_p1u32_u32_p1i8(
    uint32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_ascending_p1u32_u32_p3i8(
    uint32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_descending_p1u32_u32_p1i8(
    uint32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_descending_p1u32_u32_p3i8(
    uint32_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_ascending_p1u64_u32_p1i8(
    uint64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_ascending_p1u64_u32_p3i8(
    uint64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_descending_p1u64_u32_p1i8(
    uint64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_descending_p1u64_u32_p3i8(
    uint64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_ascending_p1u64_u32_p1i8(
    uint64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_ascending_p1u64_u32_p3i8(
    uint64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_descending_p1u64_u32_p1i8(
    uint64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_descending_p1u64_u32_p3i8(
    uint64_t *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<uint64_t>{});
}

//================= default work grop private sort for fp32 ====================

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_ascending_p1f32_u32_p1i8(
    float *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<float>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_ascending_p1f32_u32_p3i8(
    float *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::less<float>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_descending_p1f32_u32_p1i8(
    float *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<float>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_descending_p1f32_u32_p3i8(
    float *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch, std::greater<float>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_ascending_p1f32_u32_p1i8(
    float *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<float>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_ascending_p1f32_u32_p3i8(
    float *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::less<float>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_descending_p1f32_u32_p1i8(
    float *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<float>{});
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_descending_p1f32_u32_p3i8(
    float *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch, std::greater<float>{});
}

//================= default work grop private sort for fp16 ====================

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_ascending_p1f16_u32_p1i8(
    _Float16 *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch,
                           [](_Float16 a, _Float16 b) { return (a < b); });
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_ascending_p1f16_u32_p3i8(
    _Float16 *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch,
                           [](_Float16 a, _Float16 b) { return (a < b); });
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_descending_p1f16_u32_p1i8(
    _Float16 *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch,
                           [](_Float16 a, _Float16 b) { return (a > b); });
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_close_descending_p1f16_u32_p3i8(
    _Float16 *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_close(first, n, scratch,
                           [](_Float16 a, _Float16 b) { return (a > b); });
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_ascending_p1f16_u32_p1i8(
    _Float16 *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch,
                            [](_Float16 a, _Float16 b) { return (a < b); });
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_ascending_p1f16_u32_p3i8(
    _Float16 *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch,
                            [](_Float16 a, _Float16 b) { return (a < b); });
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_descending_p1f16_u32_p1i8(
    _Float16 *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch,
                            [](_Float16 a, _Float16 b) { return (a > b); });
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_private_sort_spread_descending_p1f16_u32_p3i8(
    _Float16 *first, uint32_t n, uint8_t *scratch) {
  private_merge_sort_spread(first, n, scratch,
                            [](_Float16 a, _Float16 b) { return (a > b); });
}

//============= default sub group private sort for signed integer =============
DEVICE_EXTERN_C_INLINE
int8_t
__devicelib_default_sub_group_private_sort_ascending_i8(int8_t value,
                                                        uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
int16_t
__devicelib_default_sub_group_private_sort_ascending_i16(int16_t value,
                                                         uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
int32_t
__devicelib_default_sub_group_private_sort_ascending_i32(int32_t value,
                                                         uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
int64_t
__devicelib_default_sub_group_private_sort_ascending_i64(int64_t value,
                                                         uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
uint8_t
__devicelib_default_sub_group_private_sort_ascending_u8(uint8_t value,
                                                        uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
uint16_t
__devicelib_default_sub_group_private_sort_ascending_u16(uint16_t value,
                                                         uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
uint32_t
__devicelib_default_sub_group_private_sort_ascending_u32(uint32_t value,
                                                         uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
uint64_t
__devicelib_default_sub_group_private_sort_ascending_u64(uint64_t value,
                                                         uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
float __devicelib_default_sub_group_private_sort_ascending_f32(
    float value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::less<float>{});
}

DEVICE_EXTERN_C_INLINE
_Float16
__devicelib_default_sub_group_private_sort_ascending_f16(_Float16 value,
                                                         uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch,
                              [](_Float16 a, _Float16 b) { return (a < b); });
}

DEVICE_EXTERN_C_INLINE
int8_t
__devicelib_default_sub_group_private_sort_descending_i8(int8_t value,
                                                         uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<int8_t>{});
}

DEVICE_EXTERN_C_INLINE
int16_t
__devicelib_default_sub_group_private_sort_descending_i16(int16_t value,
                                                          uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<int16_t>{});
}

DEVICE_EXTERN_C_INLINE
int32_t
__devicelib_default_sub_group_private_sort_descending_i32(int32_t value,
                                                          uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<int32_t>{});
}

DEVICE_EXTERN_C_INLINE
int64_t
__devicelib_default_sub_group_private_sort_descending_i64(int64_t value,
                                                          uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<int64_t>{});
}

DEVICE_EXTERN_C_INLINE
uint8_t
__devicelib_default_sub_group_private_sort_descending_u8(uint8_t value,
                                                         uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<uint8_t>{});
}

DEVICE_EXTERN_C_INLINE
uint16_t
__devicelib_default_sub_group_private_sort_descending_u16(uint16_t value,
                                                          uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<uint16_t>{});
}

DEVICE_EXTERN_C_INLINE
uint32_t
__devicelib_default_sub_group_private_sort_descending_u32(uint32_t value,
                                                          uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<uint32_t>{});
}

DEVICE_EXTERN_C_INLINE
uint64_t
__devicelib_default_sub_group_private_sort_descending_u64(uint64_t value,
                                                          uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<uint64_t>{});
}

DEVICE_EXTERN_C_INLINE
float __devicelib_default_sub_group_private_sort_descending_f32(
    float value, uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch, std::greater<float>{});
}

DEVICE_EXTERN_C_INLINE
_Float16
__devicelib_default_sub_group_private_sort_descending_f16(_Float16 value,
                                                          uint8_t *scratch) {
  return sub_group_merge_sort(value, scratch,
                              [](_Float16 a, _Float16 b) { return (a > b); });
}

#endif
