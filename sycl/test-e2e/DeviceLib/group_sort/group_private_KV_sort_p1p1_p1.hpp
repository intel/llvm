#include "group_sort.hpp"

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_ascending_p1u32_p1u32_u32_p1i8(
    uint32_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

__DPCPP_SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_private_sort_close_descending_p1u32_p1u32_u32_p1i8(
    uint32_t *keys, uint32_t *vals, uint32_t n, uint8_t *scratch);

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
