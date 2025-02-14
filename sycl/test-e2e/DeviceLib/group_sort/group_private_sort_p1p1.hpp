#pragma once
#include <sycl/sycl.hpp>
#ifdef __SYCL_DEVICE_ONLY__
SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i8_u32_p1i8(
    int8_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i8_u32_p1i8(
    int8_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i8_u32_p1i8(
    int8_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i8_u32_p1i8(
    int8_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i16_u32_p1i8(
    int16_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i16_u32_p1i8(
    int16_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i16_u32_p1i8(
    int16_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i16_u32_p1i8(
    int16_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch);
#else
extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i8_u32_p1i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i8_u32_p1i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i8_u32_p1i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i8_u32_p1i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i16_u32_p1i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i16_u32_p1i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i16_u32_p1i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i16_u32_p1i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

#endif
