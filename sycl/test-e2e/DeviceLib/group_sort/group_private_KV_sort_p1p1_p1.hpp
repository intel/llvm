#include "group_sort.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sycl/sycl.hpp>
#include <tuple>
#include <vector>
__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u8_p1u8_u32_p1i8(
    uint8_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u8_p1u8_u32_p1i8(
    uint8_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u8_p1i8_u32_p1i8(
    uint8_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u8_p1i8_u32_p1i8(
    uint8_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u8_p1u16_u32_p1i8(
    uint8_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u8_p1u16_u32_p1i8(
    uint8_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u8_p1i16_u32_p1i8(
    uint8_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u8_p1i16_u32_p1i8(
    uint8_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u8_p1u32_u32_p1i8(
    uint8_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u8_p1u32_u32_p1i8(
    uint8_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u8_p1i32_u32_p1i8(
    uint8_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u8_p1i32_u32_p1i8(
    uint8_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u8_p1u64_u32_p1i8(
    uint8_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u8_p1u64_u32_p1i8(
    uint8_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u8_p1i64_u32_p1i8(
    uint8_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u8_p1i64_u32_p1i8(
    uint8_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u8_p1f32_u32_p1i8(
    uint8_t *keys, float *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i8_p1u8_u32_p1i8(
    int8_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i8_p1u8_u32_p1i8(
    int8_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i8_p1i8_u32_p1i8(
    int8_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i8_p1i8_u32_p1i8(
    int8_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i8_p1u16_u32_p1i8(
    int8_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i8_p1u16_u32_p1i8(
    int8_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i8_p1i16_u32_p1i8(
    int8_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i8_p1i16_u32_p1i8(
    int8_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i8_p1u32_u32_p1i8(
    int8_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i8_p1u32_u32_p1i8(
    int8_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i8_p1i32_u32_p1i8(
    int8_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i8_p1i32_u32_p1i8(
    int8_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i8_p1u64_u32_p1i8(
    int8_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i8_p1u64_u32_p1i8(
    int8_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i8_p1i64_u32_p1i8(
    int8_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i8_p1i64_u32_p1i8(
    int8_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i8_p1f32_u32_p1i8(
    int8_t *keys, float *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u16_p1u8_u32_p1i8(
    uint16_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u16_p1u8_u32_p1i8(
    uint16_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u16_p1i8_u32_p1i8(
    uint16_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u16_p1i8_u32_p1i8(
    uint16_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u16_p1u16_u32_p1i8(
    uint16_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u16_p1u16_u32_p1i8(
    uint16_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u16_p1i16_u32_p1i8(
    uint16_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u16_p1i16_u32_p1i8(
    uint16_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u16_p1u32_u32_p1i8(
    uint16_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u16_p1u32_u32_p1i8(
    uint16_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u16_p1i32_u32_p1i8(
    uint16_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u16_p1i32_u32_p1i8(
    uint16_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u16_p1u64_u32_p1i8(
    uint16_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u16_p1u64_u32_p1i8(
    uint16_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u16_p1i64_u32_p1i8(
    uint16_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u16_p1i64_u32_p1i8(
    uint16_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u16_p1f32_u32_p1i8(
    uint16_t *keys, float *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u16_p1f32_u32_p1i8(
    uint16_t *keys, float *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i16_p1u8_u32_p1i8(
    int16_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i16_p1u8_u32_p1i8(
    int16_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i16_p1i8_u32_p1i8(
    int16_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i16_p1i8_u32_p1i8(
    int16_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i16_p1u16_u32_p1i8(
    int16_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i16_p1u16_u32_p1i8(
    int16_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i16_p1i16_u32_p1i8(
    int16_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i16_p1i16_u32_p1i8(
    int16_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i16_p1u32_u32_p1i8(
    int16_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i16_p1u32_u32_p1i8(
    int16_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i16_p1i32_u32_p1i8(
    int16_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i16_p1i32_u32_p1i8(
    int16_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i16_p1u64_u32_p1i8(
    int16_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i16_p1u64_u32_p1i8(
    int16_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i16_p1i64_u32_p1i8(
    int16_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i16_p1i64_u32_p1i8(
    int16_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i16_p1f32_u32_p1i8(
    int16_t *keys, float *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i16_p1f32_u32_p1i8(
    int16_t *keys, float *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u32_p1u8_u32_p1i8(
    uint32_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u32_p1u8_u32_p1i8(
    uint32_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u32_p1i8_u32_p1i8(
    uint32_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u32_p1i8_u32_p1i8(
    uint32_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u32_p1u16_u32_p1i8(
    uint32_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u32_p1u16_u32_p1i8(
    uint32_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u32_p1i16_u32_p1i8(
    uint32_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u32_p1i16_u32_p1i8(
    uint32_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u32_p1u32_u32_p1i8(
    uint32_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u32_p1u32_u32_p1i8(
    uint32_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u32_p1i32_u32_p1i8(
    uint32_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u32_p1i32_u32_p1i8(
    uint32_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u32_p1u64_u32_p1i8(
    uint32_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u32_p1u64_u32_p1i8(
    uint32_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u32_p1i64_u32_p1i8(
    uint32_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u32_p1i64_u32_p1i8(
    uint32_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i32_p1u8_u32_p1i8(
    int32_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i32_p1u8_u32_p1i8(
    int32_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i32_p1i8_u32_p1i8(
    int32_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i32_p1i8_u32_p1i8(
    int32_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i32_p1u16_u32_p1i8(
    int32_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i32_p1u16_u32_p1i8(
    int32_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i32_p1i16_u32_p1i8(
    int32_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i32_p1i16_u32_p1i8(
    int32_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i32_p1u32_u32_p1i8(
    int32_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i32_p1u32_u32_p1i8(
    int32_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i32_p1i32_u32_p1i8(
    int32_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i32_p1i32_u32_p1i8(
    int32_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i32_p1u64_u32_p1i8(
    int32_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i32_p1u64_u32_p1i8(
    int32_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i32_p1i64_u32_p1i8(
    int32_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i32_p1i64_u32_p1i8(
    int32_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u64_p1u8_u32_p1i8(
    uint64_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u64_p1u8_u32_p1i8(
    uint64_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u64_p1i8_u32_p1i8(
    uint64_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u64_p1i8_u32_p1i8(
    uint64_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u64_p1u16_u32_p1i8(
    uint64_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u64_p1u16_u32_p1i8(
    uint64_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u64_p1i16_u32_p1i8(
    uint64_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u64_p1i16_u32_p1i8(
    uint64_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u64_p1u32_u32_p1i8(
    uint64_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u64_p1u32_u32_p1i8(
    uint64_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u64_p1i32_u32_p1i8(
    uint64_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u64_p1i32_u32_p1i8(
    uint64_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u64_p1u64_u32_p1i8(
    uint64_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u64_p1u64_u32_p1i8(
    uint64_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u64_p1i64_u32_p1i8(
    uint64_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u64_p1i64_u32_p1i8(
    uint64_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i64_p1u8_u32_p1i8(
    int64_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i64_p1u8_u32_p1i8(
    int64_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i64_p1i8_u32_p1i8(
    int64_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i64_p1i8_u32_p1i8(
    int64_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i64_p1u16_u32_p1i8(
    int64_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i64_p1u16_u32_p1i8(
    int64_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i64_p1i16_u32_p1i8(
    int64_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i64_p1i16_u32_p1i8(
    int64_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i64_p1u32_u32_p1i8(
    int64_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i64_p1u32_u32_p1i8(
    int64_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i64_p1i32_u32_p1i8(
    int64_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i64_p1i32_u32_p1i8(
    int64_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i64_p1u64_u32_p1i8(
    int64_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i64_p1u64_u32_p1i8(
    int64_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1i64_p1i64_u32_p1i8(
    int64_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1i64_p1i64_u32_p1i8(
    int64_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u8_p1u8_u32_p1i8(
    uint8_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u8_p1u8_u32_p1i8(
    uint8_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u8_p1i8_u32_p1i8(
    uint8_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u8_p1i8_u32_p1i8(
    uint8_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u8_p1u16_u32_p1i8(
    uint8_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u8_p1u16_u32_p1i8(
    uint8_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u8_p1i16_u32_p1i8(
    uint8_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u8_p1i16_u32_p1i8(
    uint8_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u8_p1u32_u32_p1i8(
    uint8_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u8_p1u32_u32_p1i8(
    uint8_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u8_p1i32_u32_p1i8(
    uint8_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u8_p1i32_u32_p1i8(
    uint8_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u8_p1u64_u32_p1i8(
    uint8_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u8_p1u64_u32_p1i8(
    uint8_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u8_p1i64_u32_p1i8(
    uint8_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u8_p1i64_u32_p1i8(
    uint8_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u8_p1f32_u32_p1i8(
    uint8_t *keys, float *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i8_p1u8_u32_p1i8(
    int8_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i8_p1u8_u32_p1i8(
    int8_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i8_p1i8_u32_p1i8(
    int8_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i8_p1i8_u32_p1i8(
    int8_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i8_p1u16_u32_p1i8(
    int8_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i8_p1u16_u32_p1i8(
    int8_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i8_p1i16_u32_p1i8(
    int8_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i8_p1i16_u32_p1i8(
    int8_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i8_p1u32_u32_p1i8(
    int8_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i8_p1u32_u32_p1i8(
    int8_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i8_p1i32_u32_p1i8(
    int8_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i8_p1i32_u32_p1i8(
    int8_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i8_p1u64_u32_p1i8(
    int8_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i8_p1u64_u32_p1i8(
    int8_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i8_p1i64_u32_p1i8(
    int8_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i8_p1i64_u32_p1i8(
    int8_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i8_p1f32_u32_p1i8(
    int8_t *keys, float *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u16_p1u8_u32_p1i8(
    uint16_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u16_p1u8_u32_p1i8(
    uint16_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u16_p1i8_u32_p1i8(
    uint16_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u16_p1i8_u32_p1i8(
    uint16_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u16_p1u16_u32_p1i8(
    uint16_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u16_p1u16_u32_p1i8(
    uint16_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u16_p1i16_u32_p1i8(
    uint16_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u16_p1i16_u32_p1i8(
    uint16_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u16_p1u32_u32_p1i8(
    uint16_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u16_p1u32_u32_p1i8(
    uint16_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u16_p1i32_u32_p1i8(
    uint16_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u16_p1i32_u32_p1i8(
    uint16_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u16_p1u64_u32_p1i8(
    uint16_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u16_p1u64_u32_p1i8(
    uint16_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u16_p1i64_u32_p1i8(
    uint16_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u16_p1i64_u32_p1i8(
    uint16_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u16_p1f32_u32_p1i8(
    uint16_t *keys, float *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u16_p1f32_u32_p1i8(
    uint16_t *keys, float *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i16_p1u8_u32_p1i8(
    int16_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i16_p1u8_u32_p1i8(
    int16_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i16_p1i8_u32_p1i8(
    int16_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i16_p1i8_u32_p1i8(
    int16_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i16_p1u16_u32_p1i8(
    int16_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i16_p1u16_u32_p1i8(
    int16_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i16_p1i16_u32_p1i8(
    int16_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i16_p1i16_u32_p1i8(
    int16_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i16_p1u32_u32_p1i8(
    int16_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i16_p1u32_u32_p1i8(
    int16_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i16_p1i32_u32_p1i8(
    int16_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i16_p1i32_u32_p1i8(
    int16_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i16_p1u64_u32_p1i8(
    int16_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i16_p1u64_u32_p1i8(
    int16_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i16_p1i64_u32_p1i8(
    int16_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i16_p1i64_u32_p1i8(
    int16_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i16_p1f32_u32_p1i8(
    int16_t *keys, float *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i16_p1f32_u32_p1i8(
    int16_t *keys, float *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u32_p1u8_u32_p1i8(
    uint32_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u32_p1u8_u32_p1i8(
    uint32_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u32_p1i8_u32_p1i8(
    uint32_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u32_p1i8_u32_p1i8(
    uint32_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u32_p1u16_u32_p1i8(
    uint32_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u32_p1u16_u32_p1i8(
    uint32_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u32_p1i16_u32_p1i8(
    uint32_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u32_p1i16_u32_p1i8(
    uint32_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u32_p1u32_u32_p1i8(
    uint32_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u32_p1u32_u32_p1i8(
    uint32_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u32_p1i32_u32_p1i8(
    uint32_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u32_p1i32_u32_p1i8(
    uint32_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u32_p1u64_u32_p1i8(
    uint32_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u32_p1u64_u32_p1i8(
    uint32_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u32_p1i64_u32_p1i8(
    uint32_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u32_p1i64_u32_p1i8(
    uint32_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i32_p1u8_u32_p1i8(
    int32_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i32_p1u8_u32_p1i8(
    int32_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i32_p1i8_u32_p1i8(
    int32_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i32_p1i8_u32_p1i8(
    int32_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i32_p1u16_u32_p1i8(
    int32_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i32_p1u16_u32_p1i8(
    int32_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i32_p1i16_u32_p1i8(
    int32_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i32_p1i16_u32_p1i8(
    int32_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i32_p1u32_u32_p1i8(
    int32_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i32_p1u32_u32_p1i8(
    int32_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i32_p1i32_u32_p1i8(
    int32_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i32_p1i32_u32_p1i8(
    int32_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i32_p1u64_u32_p1i8(
    int32_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i32_p1u64_u32_p1i8(
    int32_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i32_p1i64_u32_p1i8(
    int32_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i32_p1i64_u32_p1i8(
    int32_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u64_p1u8_u32_p1i8(
    uint64_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u64_p1u8_u32_p1i8(
    uint64_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u64_p1i8_u32_p1i8(
    uint64_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u64_p1i8_u32_p1i8(
    uint64_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u64_p1u16_u32_p1i8(
    uint64_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u64_p1u16_u32_p1i8(
    uint64_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u64_p1i16_u32_p1i8(
    uint64_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u64_p1i16_u32_p1i8(
    uint64_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u64_p1u32_u32_p1i8(
    uint64_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u64_p1u32_u32_p1i8(
    uint64_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u64_p1i32_u32_p1i8(
    uint64_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u64_p1i32_u32_p1i8(
    uint64_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u64_p1u64_u32_p1i8(
    uint64_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u64_p1u64_u32_p1i8(
    uint64_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1u64_p1i64_u32_p1i8(
    uint64_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1u64_p1i64_u32_p1i8(
    uint64_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i64_p1u8_u32_p1i8(
    int64_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i64_p1u8_u32_p1i8(
    int64_t *keys, uint8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i64_p1i8_u32_p1i8(
    int64_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i64_p1i8_u32_p1i8(
    int64_t *keys, int8_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i64_p1u16_u32_p1i8(
    int64_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i64_p1u16_u32_p1i8(
    int64_t *keys, uint16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i64_p1i16_u32_p1i8(
    int64_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i64_p1i16_u32_p1i8(
    int64_t *keys, int16_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i64_p1u32_u32_p1i8(
    int64_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i64_p1u32_u32_p1i8(
    int64_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i64_p1i32_u32_p1i8(
    int64_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i64_p1i32_u32_p1i8(
    int64_t *keys, int32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i64_p1u64_u32_p1i8(
    int64_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i64_p1u64_u32_p1i8(
    int64_t *keys, uint64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_ascending_p1i64_p1i64_u32_p1i8(
    int64_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_spread_descending_p1i64_p1i64_u32_p1i8(
    int64_t *keys, int64_t *vals, uint32_t n, uint8_t *scratch);

using namespace sycl;

template <typename KeyT, typename ValT, size_t WG_SZ, size_t NUM,
          typename SortHelper>
void test_work_group_KV_private_sort(sycl::queue &q, const KeyT keys[NUM],
                                     const ValT vals[NUM], SortHelper gsh) {
  static_assert((NUM % WG_SZ == 0),
                "Input number must be divisible by work group size!");

  KeyT input_keys[NUM];
  ValT input_vals[NUM];
  memcpy(&input_keys[0], &keys[0], NUM * sizeof(KeyT));
  memcpy(&input_vals[0], &vals[0], NUM * sizeof(ValT));
  size_t scratch_size = 2 * NUM * (sizeof(KeyT) + sizeof(ValT)) +
                        std::max(alignof(KeyT), alignof(ValT));
  uint8_t *scratch_ptr =
      (uint8_t *)aligned_alloc_device(alignof(KeyT), scratch_size, q);
  const static size_t wg_size = WG_SZ;
  constexpr size_t num_per_work_item = NUM / WG_SZ;
  KeyT output_keys[NUM];
  ValT output_vals[NUM];
  std::vector<std::tuple<KeyT, ValT>> sorted_vec;
  for (size_t idx = 0; idx < NUM; ++idx)
    sorted_vec.push_back(std::make_tuple(input_keys[idx], input_vals[idx]));
#ifdef DES
  auto kv_tuple_comp = [](const std::tuple<KeyT, ValT> &t1,
                          const std::tuple<KeyT, ValT> &t2) {
    return std::get<0>(t1) > std::get<0>(t2);
  };
#else
  auto kv_tuple_comp = [](const std::tuple<KeyT, ValT> &t1,
                          const std::tuple<KeyT, ValT> &t2) {
    return std::get<0>(t1) < std::get<0>(t2);
  };
#endif
  std::stable_sort(sorted_vec.begin(), sorted_vec.end(), kv_tuple_comp);

  /*for (size_t idx = 0; idx < NUM; ++idx) {
    std::cout << "key: " << (int)std::get<0>(sorted_vec[idx]) << " val: " <<
  (int)std::get<1>(sorted_vec[idx]) << std::endl;
  }*/

  nd_range<1> num_items((range<1>(wg_size)), (range<1>(wg_size)));
  {
    buffer<KeyT, 1> ikeys_buf(input_keys, NUM);
    buffer<ValT, 1> ivals_buf(input_vals, NUM);
    buffer<KeyT, 1> okeys_buf(output_keys, NUM);
    buffer<ValT, 1> ovals_buf(output_vals, NUM);
    q.submit([&](auto &h) {
       accessor ikeys_acc{ikeys_buf, h};
       accessor ivals_acc{ivals_buf, h};
       accessor okeys_acc{okeys_buf, h};
       accessor ovals_acc{ovals_buf, h};
       h.parallel_for(num_items, [=](nd_item<1> i) {
         KeyT pkeys[num_per_work_item];
         ValT pvals[num_per_work_item];
         // copy from global input to fix-size private array.
         for (size_t idx = 0; idx < num_per_work_item; ++idx) {
           pkeys[idx] =
               ikeys_acc[i.get_local_linear_id() * num_per_work_item + idx];
           pvals[idx] =
               ivals_acc[i.get_local_linear_id() * num_per_work_item + idx];
         }

         gsh(pkeys, pvals, num_per_work_item, scratch_ptr);

         for (size_t idx = 0; idx < num_per_work_item; ++idx) {
           okeys_acc[i.get_local_linear_id() * num_per_work_item + idx] =
               pkeys[idx];
           ovals_acc[i.get_local_linear_id() * num_per_work_item + idx] =
               pvals[idx];
         }
       });
     }).wait();
  }

  /* for (size_t idx = 0; idx < NUM; ++idx) {
    std::cout << "key: " << (int)(input_keys[idx]) << " val: " <<
  (int)(input_vals[idx]) << std::endl;
  }*/

  sycl::free(scratch_ptr, q);
  bool fails = false;
#ifdef SPREAD
  for (size_t idx = 0; idx < NUM; ++idx) {
    size_t idx1 = idx / WG_SZ;
    size_t idx2 = idx % WG_SZ;
    if ((output_keys[idx2 * num_per_work_item + idx1] != std::get<0>(sorted_vec[idx])) ||
        (output_vals[idx2 * num_per_work_item + idx1] != std::get<1>(sorted_vec[idx]))) {
      std::cout << "idx: " << idx << std::endl;
      fails = true;
      break;
    }
  }
#else
  for (size_t idx = 0; idx < NUM; ++idx) {
    if ((output_keys[idx] != std::get<0>(sorted_vec[idx])) ||
        (output_vals[idx] != std::get<1>(sorted_vec[idx]))) {
      std::cout << "idx: " << idx << std::endl;
      fails = true;
      break;
    }
  }
#endif
  assert(!fails);
}
