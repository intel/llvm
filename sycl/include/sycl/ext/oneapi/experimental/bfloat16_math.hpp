//==-------- bfloat16_math.hpp - SYCL bloat16 math functions ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/builtins.hpp>            // for ceil, cos, exp, exp10, exp2
#include <sycl/builtins_utils_vec.hpp>  // For simplify_if_swizzle, is_swizzle
#include <sycl/detail/memcpy.hpp>       // sycl::detail::memcpy
#include <sycl/ext/oneapi/bfloat16.hpp> // for bfloat16, bfloat16ToBits
#include <sycl/marray.hpp>              // for marray

#include <cstring>     // for size_t
#include <stdint.h>    // for uint32_t
#include <type_traits> // for enable_if_t, is_same

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

namespace detail {
template <size_t N>
uint32_t to_uint32_t(sycl::marray<bfloat16, N> x, size_t start) {
  uint32_t res;
  sycl::detail::memcpy(&res, &x[start], sizeof(uint32_t));
  return res;
}
} // namespace detail

// Trait to check if the type is a vector or swizzle of bfloat16.
template <typename T>
constexpr bool is_vec_or_swizzle_bf16_v =
    sycl::detail::is_vec_or_swizzle_v<T> &&
    sycl::detail::is_valid_elem_type_v<T, bfloat16>;

template <typename T>
constexpr int num_elements_v = sycl::detail::num_elements<T>::value;

/******************* isnan ********************/

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

// Overload for BF16 vec and swizzles.
template <typename T, int N = num_elements_v<T>>
std::enable_if_t<is_vec_or_swizzle_bf16_v<T>, sycl::vec<int16_t, N>>
isnan(T x) {

#if defined(__SYCL_DEVICE_ONLY__) && (defined(__SPIR__) || defined(__SPIRV__))
  // Convert BFloat16 vector to float vec and call isnan().
  sycl::vec<float, N> FVec =
      x.template convert<float, sycl::rounding_mode::automatic>();
  auto Res = isnan(FVec);

  // For vec<float>, the return type of isnan is vec<int32_t> so,
  // an explicit conversion is required to vec<int16_t>.
  return Res.template convert<int16_t>();
#else

  sycl::vec<int16_t, N> res;
  for (size_t i = 0; i < N; i++) {
    // The result of isnan is 0 or 1 but SPEC requires
    // isnan() of vec/swizzle to return -1 or 0.
    res[i] = isnan(x[i]) ? -1 : 0;
  }
  return res;
#endif
}

/******************* fabs ********************/

template <typename T>
std::enable_if_t<std::is_same_v<T, bfloat16>, T> fabs(T x) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&                     \
    (__SYCL_CUDA_ARCH__ >= 800)
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
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&
       // (__SYCL_CUDA_ARCH__ >= 800)
}

template <size_t N>
sycl::marray<bfloat16, N> fabs(sycl::marray<bfloat16, N> x) {
  sycl::marray<bfloat16, N> res;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&                     \
    (__SYCL_CUDA_ARCH__ >= 800)
  for (size_t i = 0; i < N / 2; i++) {
    auto partial_res = __clc_fabs(detail::to_uint32_t(x, i * 2));
    sycl::detail::memcpy(&res[i * 2], &partial_res, sizeof(uint32_t));
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
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&
       // (__SYCL_CUDA_ARCH__ >= 800)
  return res;
}

// Overload for BF16 vec and swizzles.
template <typename T, int N = num_elements_v<T>>
std::enable_if_t<is_vec_or_swizzle_bf16_v<T>, sycl::vec<bfloat16, N>>
fabs(T x) {
#if defined(__SYCL_DEVICE_ONLY__) && (defined(__SPIR__) || defined(__SPIRV__))
  // Convert BFloat16 vector to float vec.
  sycl::vec<float, N> FVec =
      x.template convert<float, sycl::rounding_mode::automatic>();
  auto Res = fabs(FVec);
  return Res.template convert<bfloat16>();
#else
  sycl::vec<bfloat16, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = fabs(x[i]);
  }
  return res;
#endif
}

/******************* fmin ********************/

template <typename T>
std::enable_if_t<std::is_same_v<T, bfloat16>, T> fmin(T x, T y) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&                     \
    (__SYCL_CUDA_ARCH__ >= 800)
  oneapi::detail::Bfloat16StorageT XBits = oneapi::detail::bfloat16ToBits(x);
  oneapi::detail::Bfloat16StorageT YBits = oneapi::detail::bfloat16ToBits(y);
  return oneapi::detail::bitsToBfloat16(__clc_fmin(XBits, YBits));
#else
  static const oneapi::detail::Bfloat16StorageT CanonicalNan = 0x7FC0;
  if (isnan(x) && isnan(y))
    return oneapi::detail::bitsToBfloat16(CanonicalNan);

  if (isnan(x))
    return y;
  if (isnan(y))
    return x;
  oneapi::detail::Bfloat16StorageT XBits = oneapi::detail::bfloat16ToBits(x);
  oneapi::detail::Bfloat16StorageT YBits = oneapi::detail::bfloat16ToBits(y);
  if (((XBits | YBits) ==
       static_cast<oneapi::detail::Bfloat16StorageT>(0x8000)) &&
      !(XBits & YBits))
    return oneapi::detail::bitsToBfloat16(
        static_cast<oneapi::detail::Bfloat16StorageT>(0x8000));

  return (x < y) ? x : y;
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&
       // (__SYCL_CUDA_ARCH__ >= 800)
}

template <size_t N>
sycl::marray<bfloat16, N> fmin(sycl::marray<bfloat16, N> x,
                               sycl::marray<bfloat16, N> y) {
  sycl::marray<bfloat16, N> res;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&                     \
    (__SYCL_CUDA_ARCH__ >= 800)
  for (size_t i = 0; i < N / 2; i++) {
    auto partial_res = __clc_fmin(detail::to_uint32_t(x, i * 2),
                                  detail::to_uint32_t(y, i * 2));
    sycl::detail::memcpy(&res[i * 2], &partial_res, sizeof(uint32_t));
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
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&
       // (__SYCL_CUDA_ARCH__ >= 800)
  return res;
}

// Overload for different combination of BF16 vec and swizzles.
template <typename T1, typename T2, int N1 = num_elements_v<T1>,
          int N2 = num_elements_v<T2>>
std::enable_if_t<is_vec_or_swizzle_bf16_v<T1> && is_vec_or_swizzle_bf16_v<T2> &&
                     N1 == N2,
                 sycl::vec<bfloat16, N1>>
fmin(T1 x, T2 y) {
#if defined(__SYCL_DEVICE_ONLY__) && (defined(__SPIR__) || defined(__SPIRV__))
  // Convert BFloat16 vectors to float vecs.
  sycl::vec<float, N1> FVecX =
      x.template convert<float, sycl::rounding_mode::automatic>();
  sycl::vec<float, N1> FVecY =
      y.template convert<float, sycl::rounding_mode::automatic>();
  auto Res = fmin(FVecX, FVecY);
  return Res.template convert<bfloat16>();
#else
  sycl::vec<bfloat16, N1> res;
  for (size_t i = 0; i < N1; i++) {
    res[i] = fmin(x[i], y[i]);
  }
  return res;
#endif
}

/******************* fmax ********************/

template <typename T>
std::enable_if_t<std::is_same_v<T, bfloat16>, T> fmax(T x, T y) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&                     \
    (__SYCL_CUDA_ARCH__ >= 800)
  oneapi::detail::Bfloat16StorageT XBits = oneapi::detail::bfloat16ToBits(x);
  oneapi::detail::Bfloat16StorageT YBits = oneapi::detail::bfloat16ToBits(y);
  return oneapi::detail::bitsToBfloat16(__clc_fmax(XBits, YBits));
#else
  static const oneapi::detail::Bfloat16StorageT CanonicalNan = 0x7FC0;
  if (isnan(x) && isnan(y))
    return oneapi::detail::bitsToBfloat16(CanonicalNan);

  if (isnan(x))
    return y;
  if (isnan(y))
    return x;
  oneapi::detail::Bfloat16StorageT XBits = oneapi::detail::bfloat16ToBits(x);
  oneapi::detail::Bfloat16StorageT YBits = oneapi::detail::bfloat16ToBits(y);
  if (((XBits | YBits) ==
       static_cast<oneapi::detail::Bfloat16StorageT>(0x8000)) &&
      !(XBits & YBits))
    return oneapi::detail::bitsToBfloat16(0);

  return (x > y) ? x : y;
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&
       // (__SYCL_CUDA_ARCH__ >= 800)
}

template <size_t N>
sycl::marray<bfloat16, N> fmax(sycl::marray<bfloat16, N> x,
                               sycl::marray<bfloat16, N> y) {
  sycl::marray<bfloat16, N> res;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&                     \
    (__SYCL_CUDA_ARCH__ >= 800)
  for (size_t i = 0; i < N / 2; i++) {
    auto partial_res = __clc_fmax(detail::to_uint32_t(x, i * 2),
                                  detail::to_uint32_t(y, i * 2));
    sycl::detail::memcpy(&res[i * 2], &partial_res, sizeof(uint32_t));
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
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&
       // (__SYCL_CUDA_ARCH__ >= 800)
  return res;
}

// Overload for different combination of BF16 vec and swizzles.
template <typename T1, typename T2, int N1 = num_elements_v<T1>,
          int N2 = num_elements_v<T2>>
std::enable_if_t<is_vec_or_swizzle_bf16_v<T1> && is_vec_or_swizzle_bf16_v<T2> &&
                     N1 == N2,
                 sycl::vec<bfloat16, N1>>
fmax(T1 x, T2 y) {
#if defined(__SYCL_DEVICE_ONLY__) && (defined(__SPIR__) || defined(__SPIRV__))
  // Convert BFloat16 vectors to float vecs.
  sycl::vec<float, N1> FVecX =
      x.template convert<float, sycl::rounding_mode::automatic>();
  sycl::vec<float, N1> FVecY =
      y.template convert<float, sycl::rounding_mode::automatic>();
  auto Res = fmax(FVecX, FVecY);
  return Res.template convert<bfloat16>();
#else
  sycl::vec<bfloat16, N1> res;
  for (size_t i = 0; i < N1; i++) {
    res[i] = fmax(x[i], y[i]);
  }
  return res;
#endif
}

/******************* fma *********************/

template <typename T>
std::enable_if_t<std::is_same_v<T, bfloat16>, T> fma(T x, T y, T z) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&                     \
    (__SYCL_CUDA_ARCH__ >= 800)
  oneapi::detail::Bfloat16StorageT XBits = oneapi::detail::bfloat16ToBits(x);
  oneapi::detail::Bfloat16StorageT YBits = oneapi::detail::bfloat16ToBits(y);
  oneapi::detail::Bfloat16StorageT ZBits = oneapi::detail::bfloat16ToBits(z);
  return oneapi::detail::bitsToBfloat16(__clc_fma(XBits, YBits, ZBits));
#else
  return sycl::ext::oneapi::bfloat16{sycl::fma(float{x}, float{y}, float{z})};
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&
       // (__SYCL_CUDA_ARCH__ >= 800)
}

template <size_t N>
sycl::marray<bfloat16, N> fma(sycl::marray<bfloat16, N> x,
                              sycl::marray<bfloat16, N> y,
                              sycl::marray<bfloat16, N> z) {
  sycl::marray<bfloat16, N> res;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&                     \
    (__SYCL_CUDA_ARCH__ >= 800)
  for (size_t i = 0; i < N / 2; i++) {
    auto partial_res =
        __clc_fma(detail::to_uint32_t(x, i * 2), detail::to_uint32_t(y, i * 2),
                  detail::to_uint32_t(z, i * 2));
    sycl::detail::memcpy(&res[i * 2], &partial_res, sizeof(uint32_t));
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
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&
       // (__SYCL_CUDA_ARCH__ >= 800)
  return res;
}

// Overload for different combination of BF16 vec and swizzles.
template <typename T1, typename T2, typename T3, int N1 = num_elements_v<T1>,
          int N2 = num_elements_v<T2>, int N3 = num_elements_v<T3>>
std::enable_if_t<is_vec_or_swizzle_bf16_v<T1> && is_vec_or_swizzle_bf16_v<T2> &&
                     is_vec_or_swizzle_bf16_v<T3> && N1 == N2 && N2 == N3,
                 sycl::vec<bfloat16, N1>>
fma(T1 x, T2 y, T3 z) {
#if defined(__SYCL_DEVICE_ONLY__) && (defined(__SPIR__) || defined(__SPIRV__))
  // Convert BFloat16 vectors to float vecs.
  sycl::vec<float, N1> FVecX =
      x.template convert<float, sycl::rounding_mode::automatic>();
  sycl::vec<float, N1> FVecY =
      y.template convert<float, sycl::rounding_mode::automatic>();
  sycl::vec<float, N1> FVecZ =
      z.template convert<float, sycl::rounding_mode::automatic>();

  auto Res = fma(FVecX, FVecY, FVecZ);
  return Res.template convert<bfloat16>();
#else
  sycl::vec<bfloat16, N1> res;
  for (size_t i = 0; i < N1; i++) {
    res[i] = fma(x[i], y[i], z[i]);
  }
  return res;
#endif
}

/******************* unary math operations ********************/

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

#if defined(__SYCL_DEVICE_ONLY__) && (defined(__SPIR__) || defined(__SPIRV__))
#define BFLOAT16_MATH_FP32_WRAPPERS_VEC(op)                                    \
  /* Overload for BF16 vec and swizzles. */                                    \
  template <typename T, int N = num_elements_v<T>>                             \
  std::enable_if_t<is_vec_or_swizzle_bf16_v<T>, sycl::vec<bfloat16, N>> op(    \
      T x) {                                                                   \
    sycl::vec<float, N> FVec =                                                 \
        x.template convert<float, sycl::rounding_mode::automatic>();           \
    auto Res = op(FVec);                                                       \
    return Res.template convert<bfloat16>();                                   \
  }
#else
#define BFLOAT16_MATH_FP32_WRAPPERS_VEC(op)                                    \
  /* Overload for BF16 vec and swizzles. */                                    \
  template <typename T, int N = num_elements_v<T>>                             \
  std::enable_if_t<is_vec_or_swizzle_bf16_v<T>, sycl::vec<bfloat16, N>> op(    \
      T x) {                                                                   \
    sycl::vec<bfloat16, N> res;                                                \
    for (size_t i = 0; i < N; i++) {                                           \
      res[i] = op(x[i]);                                                       \
    }                                                                          \
    return res;                                                                \
  }
#endif

BFLOAT16_MATH_FP32_WRAPPERS(ceil)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(ceil)
BFLOAT16_MATH_FP32_WRAPPERS_VEC(ceil)

BFLOAT16_MATH_FP32_WRAPPERS(cos)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(cos)
BFLOAT16_MATH_FP32_WRAPPERS_VEC(cos)

BFLOAT16_MATH_FP32_WRAPPERS(exp)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(exp)
BFLOAT16_MATH_FP32_WRAPPERS_VEC(exp)

BFLOAT16_MATH_FP32_WRAPPERS(exp10)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(exp10)
BFLOAT16_MATH_FP32_WRAPPERS_VEC(exp10)

BFLOAT16_MATH_FP32_WRAPPERS(exp2)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(exp2)
BFLOAT16_MATH_FP32_WRAPPERS_VEC(exp2)

BFLOAT16_MATH_FP32_WRAPPERS(floor)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(floor)
BFLOAT16_MATH_FP32_WRAPPERS_VEC(floor)

BFLOAT16_MATH_FP32_WRAPPERS(log)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(log)
BFLOAT16_MATH_FP32_WRAPPERS_VEC(log)

BFLOAT16_MATH_FP32_WRAPPERS(log2)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(log2)
BFLOAT16_MATH_FP32_WRAPPERS_VEC(log2)

BFLOAT16_MATH_FP32_WRAPPERS(log10)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(log10)
BFLOAT16_MATH_FP32_WRAPPERS_VEC(log10)

BFLOAT16_MATH_FP32_WRAPPERS(rint)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(rint)
BFLOAT16_MATH_FP32_WRAPPERS_VEC(rint)

BFLOAT16_MATH_FP32_WRAPPERS(rsqrt)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(rsqrt)
BFLOAT16_MATH_FP32_WRAPPERS_VEC(rsqrt)

BFLOAT16_MATH_FP32_WRAPPERS(sin)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(sin)
BFLOAT16_MATH_FP32_WRAPPERS_VEC(sin)

BFLOAT16_MATH_FP32_WRAPPERS(sqrt)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(sqrt)
BFLOAT16_MATH_FP32_WRAPPERS_VEC(sqrt)

BFLOAT16_MATH_FP32_WRAPPERS(trunc)
BFLOAT16_MATH_FP32_WRAPPERS_MARRAY(trunc)
BFLOAT16_MATH_FP32_WRAPPERS_VEC(trunc)

#undef BFLOAT16_MATH_FP32_WRAPPERS
#undef BFLOAT16_MATH_FP32_WRAPPERS_MARRAY
#undef BFLOAT16_MATH_FP32_WRAPPERS_VEC
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
