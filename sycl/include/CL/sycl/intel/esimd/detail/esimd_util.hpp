//==----------------- esimd_util.hpp - DPC++ Explicit SIMD API  ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utility functions used for implementing Explicit SIMD APIs.
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/type_traits.hpp>

namespace __esimd {

/// Constant in number of bytes.
enum { BYTE = 1, WORD = 2, DWORD = 4, QWORD = 8, OWORD = 16, GRF = 32 };

/// Compute the next power of 2 at compile time.
static ESIMD_INLINE constexpr unsigned int getNextPowerOf2(unsigned int n,
                                                           unsigned int k = 1) {
  return (k >= n) ? k : getNextPowerOf2(n, k * 2);
}

/// Check if a given 32 bit positive integer is a power of 2 at compile time.
static ESIMD_INLINE constexpr bool isPowerOf2(unsigned int n) {
  return (n & (n - 1)) == 0;
}

static ESIMD_INLINE constexpr bool isPowerOf2(unsigned int n,
                                              unsigned int limit) {
  return (n & (n - 1)) == 0 && n <= limit;
}

static ESIMD_INLINE constexpr unsigned log2(unsigned n) {
  return (n > 1) ? 1 + log2(n >> 1) : 0;
}

} // namespace __esimd

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace intel {
namespace gpu {

namespace details {

/// type traits
template <typename T> struct is_esimd_vector {
  static const bool value = false;
};
template <typename T, int N>
struct is_esimd_vector<sycl::intel::gpu::simd<T, N>> {
  static const bool value = true;
};
template <typename T, int N>
struct is_esimd_vector<sycl::intel::gpu::vector_type<T, N>> {
  static const bool value = true;
};

template <typename T>
struct is_esimd_scalar
    : std::integral_constant<bool, cl::sycl::detail::is_arithmetic<T>::value> {
};

template <typename T>
struct is_dword_type
    : std::integral_constant<
          bool, std::is_same<int, typename std::remove_const<T>::type>::value ||
                    std::is_same<unsigned int,
                                 typename std::remove_const<T>::type>::value> {
};

template <typename T, int N>
struct is_dword_type<sycl::intel::gpu::vector_type<T, N>> {
  static const bool value = is_dword_type<T>::value;
};

template <typename T, int N>
struct is_dword_type<sycl::intel::gpu::simd<T, N>> {
  static const bool value = is_dword_type<T>::value;
};

template <typename T>
struct is_word_type
    : std::integral_constant<
          bool,
          std::is_same<short, typename std::remove_const<T>::type>::value ||
              std::is_same<unsigned short,
                           typename std::remove_const<T>::type>::value> {};

template <typename T, int N>
struct is_word_type<sycl::intel::gpu::vector_type<T, N>> {
  static const bool value = is_word_type<T>::value;
};

template <typename T, int N> struct is_word_type<sycl::intel::gpu::simd<T, N>> {
  static const bool value = is_word_type<T>::value;
};

template <typename T>
struct is_byte_type
    : std::integral_constant<
          bool,
          std::is_same<char, typename std::remove_const<T>::type>::value ||
              std::is_same<unsigned char,
                           typename std::remove_const<T>::type>::value> {};

template <typename T, int N>
struct is_byte_type<sycl::intel::gpu::vector_type<T, N>> {
  static const bool value = is_byte_type<T>::value;
};

template <typename T, int N> struct is_byte_type<sycl::intel::gpu::simd<T, N>> {
  static const bool value = is_byte_type<T>::value;
};

template <typename T>
struct is_fp_type
    : std::integral_constant<
          bool,
          std::is_same<float, typename std::remove_const<T>::type>::value> {};

template <typename T>
struct is_df_type
    : std::integral_constant<
          bool,
          std::is_same<double, typename std::remove_const<T>::type>::value> {};

template <typename T>
struct is_fp_or_dword_type
    : std::integral_constant<
          bool,
          std::is_same<float, typename std::remove_const<T>::type>::value ||
              std::is_same<int, typename std::remove_const<T>::type>::value ||
              std::is_same<unsigned int,
                           typename std::remove_const<T>::type>::value> {};

template <typename T>
struct is_qword_type
    : std::integral_constant<
          bool,
          std::is_same<long long, typename std::remove_const<T>::type>::value ||
              std::is_same<unsigned long long,
                           typename std::remove_const<T>::type>::value> {};

template <typename T, int N>
struct is_qword_type<sycl::intel::gpu::vector_type<T, N>> {
  static const bool value = is_qword_type<T>::value;
};

template <typename T, int N>
struct is_qword_type<sycl::intel::gpu::simd<T, N>> {
  static const bool value = is_qword_type<T>::value;
};

// Extends to ESIMD vector types.
template <typename T, int N>
struct is_fp_or_dword_type<sycl::intel::gpu::vector_type<T, N>> {
  static const bool value = is_fp_or_dword_type<T>::value;
};

template <typename T, int N>
struct is_fp_or_dword_type<sycl::intel::gpu::simd<T, N>> {
  static const bool value = is_fp_or_dword_type<T>::value;
};

/// Convert types into vector types
template <typename T> struct simd_type {
  using type = sycl::intel::gpu::simd<T, 1>;
};
template <typename T, int N>
struct simd_type<sycl::intel::gpu::vector_type<T, N>> {
  using type = sycl::intel::gpu::simd<T, N>;
};

template <typename T> struct simd_type<T &> {
  using type = typename simd_type<T>::type;
};
template <typename T> struct simd_type<T &&> {
  using type = typename simd_type<T>::type;
};
template <typename T> struct simd_type<const T> {
  using type = typename simd_type<T>::type;
};

template <typename T> struct dword_type { using type = T; };
template <> struct dword_type<char> { using type = int; };
template <> struct dword_type<short> { using type = int; };
template <> struct dword_type<uchar> { using type = uint; };
template <> struct dword_type<ushort> { using type = uint; };

template <typename T> struct byte_type { using type = T; };
template <> struct byte_type<short> { using type = char; };
template <> struct byte_type<int> { using type = char; };
template <> struct byte_type<ushort> { using type = uchar; };
template <> struct byte_type<uint> { using type = uchar; };

template <typename T> struct word_type { using type = T; };
template <> struct word_type<char> { using type = short; };
template <> struct word_type<int> { using type = short; };
template <> struct word_type<uchar> { using type = ushort; };
template <> struct word_type<uint> { using type = ushort; };

} // namespace details
} // namespace gpu
} // namespace intel
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
