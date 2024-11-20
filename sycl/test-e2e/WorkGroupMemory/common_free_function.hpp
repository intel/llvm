#pragma once

#include "common.hpp"
#include <cassert>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<1>))
void sum(sycl::ext::oneapi::experimental::work_group_memory<T[]> mem, T *buf,
         sycl::ext::oneapi::experimental::work_group_memory<T> result,
         T expected, size_t WGSIZE, bool UseHelper) {
  const auto it = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  size_t local_id = it.get_local_id();
  mem[local_id] = buf[local_id];
  group_barrier(it.get_group());
  if (it.get_group().leader()) {
    result = 0;
    if (!UseHelper) {
      for (int i = 0; i < WGSIZE; ++i) {
        result = result + mem[i];
      }
    } else {
      sum_helper(mem, result, WGSIZE);
    }
    assert(result == expected);
  }
}

// Explicit instantiations for the relevant data types.
// These are needed because free function kernel support is not fully
// implemented yet.
// TODO: Remove these once free function kernel support is fully there.
#define SUM(T)                                                                 \
  template void sum<T>(                                                        \
      sycl::ext::oneapi::experimental::work_group_memory<T[]> mem, T * buf,    \
      sycl::ext::oneapi::experimental::work_group_memory<T> result,            \
      T expected, size_t WGSIZE, bool UseHelper);

SUM(int)
SUM(uint16_t)
SUM(half)
SUM(double)
SUM(float)
SUM(char)
SUM(bool)

template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<1>))
void sum_marray(
    sycl::ext::oneapi::experimental::work_group_memory<sycl::marray<T, 16>> mem,
    T *buf, sycl::ext::oneapi::experimental::work_group_memory<T> result,
    T expected) {
  const auto it = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  size_t local_id = it.get_local_id();
  constexpr T tolerance = 0.0001;
  sycl::marray<T, 16> &data = mem;
  data[local_id] = buf[local_id];
  group_barrier(it.get_group());
  if (it.get_group().leader()) {
    result = 0;
    for (int i = 0; i < 16; ++i) {
      result = result + data[i];
    }
    assert((result - expected) * (result - expected) <= tolerance);
  }
}

// Explicit instantiations for the relevant data types.
#define SUM_MARRAY(T)                                                          \
  template void sum_marray<T>(                                                 \
      sycl::ext::oneapi::experimental::work_group_memory<sycl::marray<T, 16>>  \
          mem,                                                                 \
      T * buf, sycl::ext::oneapi::experimental::work_group_memory<T> result,   \
      T expected);

SUM_MARRAY(float);
SUM_MARRAY(double);
SUM_MARRAY(half);

template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<1>))
void sum_vec(
    sycl::ext::oneapi::experimental::work_group_memory<sycl::vec<T, 16>> mem,
    T *buf, sycl::ext::oneapi::experimental::work_group_memory<T> result,
    T expected) {
  const auto it = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  size_t local_id = it.get_local_id();
  constexpr T tolerance = 0.0001;
  sycl::vec<T, 16> &data = mem;
  data[local_id] = buf[local_id];
  group_barrier(it.get_group());
  if (it.get_group().leader()) {
    result = 0;
    for (int i = 0; i < 16; ++i) {
      result = result + data[i];
    }
    assert((result - expected) * (result - expected) <= tolerance);
  }
}

// Explicit instantiations for the relevant data types.
#define SUM_VEC(T)                                                             \
  template void sum_vec<T>(                                                    \
      sycl::ext::oneapi::experimental::work_group_memory<sycl::vec<T, 16>>     \
          mem,                                                                 \
      T * buf, sycl::ext::oneapi::experimental::work_group_memory<T> result,   \
      T expected);

SUM_VEC(float);
SUM_VEC(double);
SUM_VEC(half);
