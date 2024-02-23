
//==--- fallback_gsort_fp32.cpp - fallback implementation of group sort
//-----==//
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
#endif
