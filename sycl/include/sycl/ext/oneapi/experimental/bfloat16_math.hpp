//==-------- bfloat16_math.hpp - SYCL bloat16 math functions ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/exception.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/marray.hpp>

#include <cstring>
#include <tuple>
#include <type_traits>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi::experimental {

namespace detail {
template <size_t N>
uint32_t to_uint32_t(sycl::marray<bfloat16, N> x, size_t start) {
  uint32_t res;
  std::memcpy(&res, &x[start], sizeof(uint32_t));
  return res;
}
} // namespace detail

template <typename T>
std::enable_if_t<std::is_same<T, bfloat16>::value, T> fabs(T x) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  oneapi::detail::Bfloat16StorageT XBits = oneapi::detail::bfloat16ToBits(x);
  return oneapi::detail::bitsToBfloat16(__clc_fabs(XBits));
#else
  std::ignore = x;
  throw runtime_error(
      "bfloat16 math functions are not currently supported on the host device.",
      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
}

template <size_t N>
sycl::marray<bfloat16, N> fabs(sycl::marray<bfloat16, N> x) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  sycl::marray<bfloat16, N> res;

  for (size_t i = 0; i < N / 2; i++) {
    auto partial_res = __clc_fabs(detail::to_uint32_t(x, i * 2));
    std::memcpy(&res[i * 2], &partial_res, sizeof(uint32_t));
  }

  if (N % 2) {
    oneapi::detail::Bfloat16StorageT XBits =
        oneapi::detail::bfloat16ToBits(x[N - 1]);
    res[N - 1] = oneapi::detail::bitsToBfloat16(__clc_fabs(XBits));
  }
  return res;
#else
  std::ignore = x;
  throw runtime_error(
      "bfloat16 math functions are not currently supported on the host device.",
      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
}

template <typename T>
std::enable_if_t<std::is_same<T, bfloat16>::value, T> fmin(T x, T y) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  oneapi::detail::Bfloat16StorageT XBits = oneapi::detail::bfloat16ToBits(x);
  oneapi::detail::Bfloat16StorageT YBits = oneapi::detail::bfloat16ToBits(y);
  return oneapi::detail::bitsToBfloat16(__clc_fmin(XBits, YBits));
#else
  std::ignore = x;
  std::ignore = y;
  throw runtime_error(
      "bfloat16 math functions are not currently supported on the host device.",
      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
}

template <size_t N>
sycl::marray<bfloat16, N> fmin(sycl::marray<bfloat16, N> x,
                               sycl::marray<bfloat16, N> y) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  sycl::marray<bfloat16, N> res;

  for (size_t i = 0; i < N / 2; i++) {
    auto partial_res = __clc_fmin(detail::to_uint32_t(x, i * 2),
                                  detail::to_uint32_t(y, i * 2));
    std::memcpy(&res[i * 2], &partial_res, sizeof(uint32_t));
  }

  if (N % 2) {
    oneapi::detail::Bfloat16StorageT XBits =
        oneapi::detail::bfloat16ToBits(x[N - 1]);
    oneapi::detail::Bfloat16StorageT YBits =
        oneapi::detail::bfloat16ToBits(y[N - 1]);
    res[N - 1] = oneapi::detail::bitsToBfloat16(__clc_fmin(XBits, YBits));
  }

  return res;
#else
  std::ignore = x;
  std::ignore = y;
  throw runtime_error(
      "bfloat16 math functions are not currently supported on the host device.",
      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
}

template <typename T>
std::enable_if_t<std::is_same<T, bfloat16>::value, T> fmax(T x, T y) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  oneapi::detail::Bfloat16StorageT XBits = oneapi::detail::bfloat16ToBits(x);
  oneapi::detail::Bfloat16StorageT YBits = oneapi::detail::bfloat16ToBits(y);
  return oneapi::detail::bitsToBfloat16(__clc_fmax(XBits, YBits));
#else
  std::ignore = x;
  std::ignore = y;
  throw runtime_error(
      "bfloat16 math functions are not currently supported on the host device.",
      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
}

template <size_t N>
sycl::marray<bfloat16, N> fmax(sycl::marray<bfloat16, N> x,
                               sycl::marray<bfloat16, N> y) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  sycl::marray<bfloat16, N> res;

  for (size_t i = 0; i < N / 2; i++) {
    auto partial_res = __clc_fmax(detail::to_uint32_t(x, i * 2),
                                  detail::to_uint32_t(y, i * 2));
    std::memcpy(&res[i * 2], &partial_res, sizeof(uint32_t));
  }

  if (N % 2) {
    oneapi::detail::Bfloat16StorageT XBits =
        oneapi::detail::bfloat16ToBits(x[N - 1]);
    oneapi::detail::Bfloat16StorageT YBits =
        oneapi::detail::bfloat16ToBits(y[N - 1]);
    res[N - 1] = oneapi::detail::bitsToBfloat16(__clc_fmax(XBits, YBits));
  }
  return res;
#else
  std::ignore = x;
  std::ignore = y;
  throw runtime_error(
      "bfloat16 math functions are not currently supported on the host device.",
      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
}

template <typename T>
std::enable_if_t<std::is_same<T, bfloat16>::value, T> fma(T x, T y, T z) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  oneapi::detail::Bfloat16StorageT XBits = oneapi::detail::bfloat16ToBits(x);
  oneapi::detail::Bfloat16StorageT YBits = oneapi::detail::bfloat16ToBits(y);
  oneapi::detail::Bfloat16StorageT ZBits = oneapi::detail::bfloat16ToBits(z);
  return oneapi::detail::bitsToBfloat16(__clc_fma(XBits, YBits, ZBits));
#else
  std::ignore = x;
  std::ignore = y;
  std::ignore = z;
  throw runtime_error(
      "bfloat16 math functions are not currently supported on the host device.",
      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
}

template <size_t N>
sycl::marray<bfloat16, N> fma(sycl::marray<bfloat16, N> x,
                              sycl::marray<bfloat16, N> y,
                              sycl::marray<bfloat16, N> z) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  sycl::marray<bfloat16, N> res;

  for (size_t i = 0; i < N / 2; i++) {
    auto partial_res =
        __clc_fma(detail::to_uint32_t(x, i * 2), detail::to_uint32_t(y, i * 2),
                  detail::to_uint32_t(z, i * 2));
    std::memcpy(&res[i * 2], &partial_res, sizeof(uint32_t));
  }

  if (N % 2) {
    oneapi::detail::Bfloat16StorageT XBits =
        oneapi::detail::bfloat16ToBits(x[N - 1]);
    oneapi::detail::Bfloat16StorageT YBits =
        oneapi::detail::bfloat16ToBits(y[N - 1]);
    oneapi::detail::Bfloat16StorageT ZBits =
        oneapi::detail::bfloat16ToBits(z[N - 1]);
    res[N - 1] = oneapi::detail::bitsToBfloat16(__clc_fma(XBits, YBits, ZBits));
  }
  return res;
#else
  std::ignore = x;
  std::ignore = y;
  std::ignore = z;
  throw runtime_error(
      "bfloat16 math functions are not currently supported on the host device.",
      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
}

} // namespace ext::oneapi::experimental
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
