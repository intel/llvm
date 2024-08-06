#pragma once
#include <sycl.hpp>

#ifdef __SYCL_DEVICE_ONLY__
SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u32_p1u32_u32_p1i8(
    uint32_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u8_p1u32_u32_p1i8(
    uint8_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);
#else
extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u32_p1u32_u32_p1i8(
    uint32_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch) {}

extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u8_p1u32_u32_p1i8(
    uint8_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch) {}
#endif
