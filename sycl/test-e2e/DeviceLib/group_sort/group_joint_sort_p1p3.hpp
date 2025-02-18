#pragma once
#include <sycl/sycl.hpp>

#ifdef __SYCL_DEVICE_ONLY__
SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1i32_u32_p3i8(
    int32_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1i32_u32_p3i8(
    int32_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1i16_u32_p3i8(
    int16_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1i16_u32_p3i8(
    int16_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1i64_u32_p3i8(
    int64_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1i64_u32_p3i8(
    int64_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1i8_u32_p3i8(
    int8_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1i8_u32_p3i8(
    int8_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1f32_u32_p3i8(
    float *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1f32_u32_p3i8(
    float *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1u8_u32_p3i8(
    uint8_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1u8_u32_p3i8(
    uint8_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1u16_u32_p3i8(
    uint16_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1u16_u32_p3i8(
    uint16_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1u32_u32_p3i8(
    uint32_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1u32_u32_p3i8(
    uint32_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1u64_u32_p3i8(
    uint64_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1u64_u32_p3i8(
    uint64_t *first, uint32_t n, uint8_t *scratch);

#else
extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1i32_u32_p3i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1i32_u32_p3i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1i16_u32_p3i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1i16_u32_p3i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1i64_u32_p3i8(
    int64_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1i64_u32_p3i8(
    int64_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1i8_u32_p3i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1i8_u32_p3i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1f32_u32_p3i8(
    float *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1f32_u32_p3i8(
    float *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1u8_u32_p3i8(
    uint8_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1u8_u32_p3i8(
    uint8_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1u16_u32_p3i8(
    uint16_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1u16_u32_p3i8(
    uint16_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1u32_u32_p3i8(
    uint32_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1u32_u32_p3i8(
    uint32_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1u64_u32_p3i8(
    uint64_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1u64_u32_p3i8(
    uint64_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

#endif
