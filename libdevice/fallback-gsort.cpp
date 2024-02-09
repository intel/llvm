
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
#if defined(__SPIR__)

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p1i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch);
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p1i32_u32_p3i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch);
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch);
}

DEVICE_EXTERN_C_INLINE
void __devicelib_default_work_group_joint_sort_ascending_p3i32_u32_p3i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  merge_sort(first, n, scratch);
}
#endif
