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

// According to bfloat16 format, NAN value's exponent field is 0xFF and
// significand has non-zero bits.
template <typename T>
std::enable_if_t<std::is_same_v<T, bfloat16>, bool> isnan(T x) {
  oneapi::detail::Bfloat16StorageT XBits = oneapi::detail::bfloat16ToBits(x);
  return (((XBits & 0x7F80) == 0x7F80) && (XBits & 0x7F)) ? true : false;
}

template <size_t N> sycl::marray<bool, N> isnan(sycl::marray<bfloat16, N> x) {
  sycl::marray<bool, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = isnan(x[i]);
  }
  return res;
}

template <typename T>
std::enable_if_t<std::is_same_v<T, bfloat16>, T> fabs(T x) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  oneapi::detail::Bfloat16StorageT XBits = oneapi::detail::bfloat16ToBits(x);
  return oneapi::detail::bitsToBfloat16(__clc_fabs(XBits));
#else
  if (!isnan(x)) {
    const static oneapi::detail::Bfloat16StorageT SignMask = 0x8000;
    oneapi::detail::Bfloat16StorageT XBits = oneapi::detail::bfloat16ToBits(x);
    x = ((XBits & SignMask) == SignMask)
            ? oneapi::detail::bitsToBfloat16(XBits & ~SignMask)
            : x;
  }
  return x;
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
}

template <size_t N>
sycl::marray<bfloat16, N> fabs(sycl::marray<bfloat16, N> x) {
  sycl::marray<bfloat16, N> res;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  for (size_t i = 0; i < N / 2; i++) {
    auto partial_res = __clc_fabs(detail::to_uint32_t(x, i * 2));
    std::memcpy(&res[i * 2], &partial_res, sizeof(uint32_t));
  }

  if (N % 2) {
    oneapi::detail::Bfloat16StorageT XBits =
        oneapi::detail::bfloat16ToBits(x[N - 1]);
    res[N - 1] = oneapi::detail::bitsToBfloat16(__clc_fabs(XBits));
  }
#else
  for (size_t i = 0; i < N; i++) {
    res[i] = fabs(x[i]);
  }
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  return res;
}

template <typename T>
std::enable_if_t<std::is_same_v<T, bfloat16>, T> fmin(T x, T y) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  oneapi::detail::Bfloat16StorageT XBits = oneapi::detail::bfloat16ToBits(x);
  oneapi::detail::Bfloat16StorageT YBits = oneapi::detail::bfloat16ToBits(y);
  return oneapi::detail::bitsToBfloat16(__clc_fmin(XBits, YBits));
#else
  static const oneapi::detail::Bfloat16StorageT CanonicalNan = 0x7FC0;
  oneapi::detail::Bfloat16StorageT XBits = oneapi::detail::bfloat16ToBits(x);
  oneapi::detail::Bfloat16StorageT YBits = oneapi::detail::bfloat16ToBits(y);
  if (isnan(x) && isnan(y))
    return oneapi::detail::bitsToBfloat16(CanonicalNan);

  if (isnan(x))
    return y;
  if (isnan(y))
    return x;
  if (((XBits | YBits) ==
       static_cast<oneapi::detail::Bfloat16StorageT>(0x8000)) &&
      !(XBits & YBits))
    return oneapi::detail::bitsToBfloat16(
        static_cast<oneapi::detail::Bfloat16StorageT>(0x8000));

  return (x < y) ? x : y;
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
}

template <size_t N>
sycl::marray<bfloat16, N> fmin(sycl::marray<bfloat16, N> x,
                               sycl::marray<bfloat16, N> y) {
  sycl::marray<bfloat16, N> res;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
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
#else
  for (size_t i = 0; i < N; i++) {
    res[i] = fmin(x[i], y[i]);
  }
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  return res;
}

template <typename T>
std::enable_if_t<std::is_same_v<T, bfloat16>, T> fmax(T x, T y) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  oneapi::detail::Bfloat16StorageT XBits = oneapi::detail::bfloat16ToBits(x);
  oneapi::detail::Bfloat16StorageT YBits = oneapi::detail::bfloat16ToBits(y);
  return oneapi::detail::bitsToBfloat16(__clc_fmax(XBits, YBits));
#else
  static const oneapi::detail::Bfloat16StorageT CanonicalNan = 0x7FC0;
  oneapi::detail::Bfloat16StorageT XBits = oneapi::detail::bfloat16ToBits(x);
  oneapi::detail::Bfloat16StorageT YBits = oneapi::detail::bfloat16ToBits(y);
  if (isnan(x) && isnan(y))
    return oneapi::detail::bitsToBfloat16(CanonicalNan);

  if (isnan(x))
    return y;
  if (isnan(y))
    return x;
  if (((XBits | YBits) ==
       static_cast<oneapi::detail::Bfloat16StorageT>(0x8000)) &&
      !(XBits & YBits))
    return oneapi::detail::bitsToBfloat16(0);

  return (x > y) ? x : y;
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
}

template <size_t N>
sycl::marray<bfloat16, N> fmax(sycl::marray<bfloat16, N> x,
                               sycl::marray<bfloat16, N> y) {
  sycl::marray<bfloat16, N> res;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
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
#else
  for (size_t i = 0; i < N; i++) {
    res[i] = fmax(x[i], y[i]);
  }
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  return res;
}

template <typename T>
std::enable_if_t<std::is_same_v<T, bfloat16>, T> fma(T x, T y, T z) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  oneapi::detail::Bfloat16StorageT XBits = oneapi::detail::bfloat16ToBits(x);
  oneapi::detail::Bfloat16StorageT YBits = oneapi::detail::bfloat16ToBits(y);
  oneapi::detail::Bfloat16StorageT ZBits = oneapi::detail::bfloat16ToBits(z);
  return oneapi::detail::bitsToBfloat16(__clc_fma(XBits, YBits, ZBits));
#else
  return sycl::ext::oneapi::bfloat16{sycl::fma(float{x}, float{y}, float{z})};
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
}

template <size_t N>
sycl::marray<bfloat16, N> fma(sycl::marray<bfloat16, N> x,
                              sycl::marray<bfloat16, N> y,
                              sycl::marray<bfloat16, N> z) {
  sycl::marray<bfloat16, N> res;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
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
#else
  for (size_t i = 0; i < N; i++) {
    res[i] = fma(x[i], y[i], z[i]);
  }
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  return res;
}

#define BFLOAT16_MATH_FP32_WRAPPERS(op)                                        \
  template <typename T>                                                        \
  std::enable_if_t<std::is_same<T, bfloat16>::value, T> op(T x) {              \
    return sycl::ext::oneapi::bfloat16{sycl::op(float{x})};                    \
  }

#define BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(op)                                 \
  template <size_t N>                                                          \
  sycl::marray<bfloat16, N> op(sycl::marray<bfloat16, N> x) {                  \
    sycl::marray<bfloat16, N> res;                                             \
    for (size_t i = 0; i < N; i++) {                                           \
      res[i] = op(x[i]);                                                       \
    }                                                                          \
    return res;                                                                \
  }

BFLOAT16_MATH_FP32_WRAPPERS(ceil)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(ceil)
BFLOAT16_MATH_FP32_WRAPPERS(cos)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(cos)
BFLOAT16_MATH_FP32_WRAPPERS(exp)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(exp)
BFLOAT16_MATH_FP32_WRAPPERS(exp10)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(exp10)
BFLOAT16_MATH_FP32_WRAPPERS(exp2)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(exp2)
BFLOAT16_MATH_FP32_WRAPPERS(floor)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(floor)
BFLOAT16_MATH_FP32_WRAPPERS(log)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(log)
BFLOAT16_MATH_FP32_WRAPPERS(log2)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(log2)
BFLOAT16_MATH_FP32_WRAPPERS(log10)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(log10)
BFLOAT16_MATH_FP32_WRAPPERS(rint)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(rint)
BFLOAT16_MATH_FP32_WRAPPERS(rsqrt)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(rsqrt)
BFLOAT16_MATH_FP32_WRAPPERS(sin)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(sin)
BFLOAT16_MATH_FP32_WRAPPERS(sqrt)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(sqrt)
BFLOAT16_MATH_FP32_WRAPPERS(trunc)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(trunc)

#undef BFLOAT16_MATH_FP32_WRAPPERS
#undef BFLOAT16_MATH_FP32_WRAPPERS_MARRAY
} // namespace ext::oneapi::experimental
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
