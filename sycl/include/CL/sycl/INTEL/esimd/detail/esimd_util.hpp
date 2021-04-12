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

#include <CL/sycl/INTEL/esimd/detail/esimd_types.hpp>
#include <CL/sycl/detail/type_traits.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace INTEL {
namespace gpu {
namespace detail {

/// ESIMD intrinsic operand size in bytes.
struct OperandSize {
  enum { BYTE = 1, WORD = 2, DWORD = 4, QWORD = 8, OWORD = 16, GRF = 32 };
};

/// Compute next power of 2 of a constexpr with guaranteed compile-time
/// evaluation.
template <unsigned int N, unsigned int K, bool K_gt_eq_N> struct NextPowerOf2;

template <unsigned int N, unsigned int K> struct NextPowerOf2<N, K, true> {
  static constexpr unsigned int get() { return K; }
};

template <unsigned int N, unsigned int K> struct NextPowerOf2<N, K, false> {
  static constexpr unsigned int get() {
    return NextPowerOf2<N, K * 2, K * 2 >= N>::get();
  }
};

template <unsigned int N> constexpr unsigned int getNextPowerOf2() {
  return NextPowerOf2<N, 1, (1 >= N)>::get();
}

template <> constexpr unsigned int getNextPowerOf2<0>() { return 0; }

/// Compute binary logarithm of a constexpr with guaranteed compile-time
/// evaluation.
template <unsigned int N, bool N_gt_1> struct Log2;

template <unsigned int N> struct Log2<N, false> {
  static constexpr unsigned int get() { return 0; }
};

template <unsigned int N> struct Log2<N, true> {
  static constexpr unsigned int get() {
    return 1 + Log2<(N >> 1), ((N >> 1) > 1)>::get();
  }
};

template <unsigned int N> constexpr unsigned int log2() {
  return Log2<N, (N > 1)>::get();
}

/// Check if a given 32 bit positive integer is a power of 2 at compile time.
static ESIMD_INLINE constexpr bool isPowerOf2(unsigned int n) {
  return (n & (n - 1)) == 0;
}

static ESIMD_INLINE constexpr bool isPowerOf2(unsigned int n,
                                              unsigned int limit) {
  return (n & (n - 1)) == 0 && n <= limit;
}

/// type traits
template <typename T> struct is_esimd_vector {
  static const bool value = false;
};
template <typename T, int N>
struct is_esimd_vector<sycl::INTEL::gpu::simd<T, N>> {
  static const bool value = true;
};
template <typename T, int N>
struct is_esimd_vector<sycl::INTEL::gpu::detail::vector_type<T, N>> {
  static const bool value = true;
};

template <typename T>
struct is_esimd_scalar
    : std::integral_constant<bool, cl::sycl::detail::is_arithmetic<T>::value> {
};

template <typename T>
struct is_dword_type
    : std::integral_constant<
          bool,
          std::is_same<int, typename sycl::detail::remove_const_t<T>>::value ||
              std::is_same<unsigned int,
                           typename sycl::detail::remove_const_t<T>>::value> {};

template <typename T, int N>
struct is_dword_type<sycl::INTEL::gpu::detail::vector_type<T, N>> {
  static const bool value = is_dword_type<T>::value;
};

template <typename T, int N>
struct is_dword_type<sycl::INTEL::gpu::simd<T, N>> {
  static const bool value = is_dword_type<T>::value;
};

template <typename T>
struct is_word_type
    : std::integral_constant<
          bool,
          std::is_same<short,
                       typename sycl::detail::remove_const_t<T>>::value ||
              std::is_same<unsigned short,
                           typename sycl::detail::remove_const_t<T>>::value> {};

template <typename T, int N>
struct is_word_type<sycl::INTEL::gpu::detail::vector_type<T, N>> {
  static const bool value = is_word_type<T>::value;
};

template <typename T, int N> struct is_word_type<sycl::INTEL::gpu::simd<T, N>> {
  static const bool value = is_word_type<T>::value;
};

template <typename T>
struct is_byte_type
    : std::integral_constant<
          bool,
          std::is_same<char, typename sycl::detail::remove_const_t<T>>::value ||
              std::is_same<unsigned char,
                           typename sycl::detail::remove_const_t<T>>::value> {};

template <typename T, int N>
struct is_byte_type<sycl::INTEL::gpu::detail::vector_type<T, N>> {
  static const bool value = is_byte_type<T>::value;
};

template <typename T, int N> struct is_byte_type<sycl::INTEL::gpu::simd<T, N>> {
  static const bool value = is_byte_type<T>::value;
};

template <typename T>
struct is_fp_type
    : std::integral_constant<
          bool, std::is_same<float,
                             typename sycl::detail::remove_const_t<T>>::value> {
};

template <typename T>
struct is_df_type
    : std::integral_constant<
          bool, std::is_same<double,
                             typename sycl::detail::remove_const_t<T>>::value> {
};

template <typename T>
struct is_fp_or_dword_type
    : std::integral_constant<
          bool,
          std::is_same<float,
                       typename sycl::detail::remove_const_t<T>>::value ||
              std::is_same<int,
                           typename sycl::detail::remove_const_t<T>>::value ||
              std::is_same<unsigned int,
                           typename sycl::detail::remove_const_t<T>>::value> {};

template <typename T>
struct is_qword_type
    : std::integral_constant<
          bool,
          std::is_same<long long,
                       typename sycl::detail::remove_const_t<T>>::value ||
              std::is_same<unsigned long long,
                           typename sycl::detail::remove_const_t<T>>::value> {};

template <typename T, int N>
struct is_qword_type<sycl::INTEL::gpu::detail::vector_type<T, N>> {
  static const bool value = is_qword_type<T>::value;
};

template <typename T, int N>
struct is_qword_type<sycl::INTEL::gpu::simd<T, N>> {
  static const bool value = is_qword_type<T>::value;
};

// Extends to ESIMD vector types.
template <typename T, int N>
struct is_fp_or_dword_type<sycl::INTEL::gpu::detail::vector_type<T, N>> {
  static const bool value = is_fp_or_dword_type<T>::value;
};

template <typename T, int N>
struct is_fp_or_dword_type<sycl::INTEL::gpu::simd<T, N>> {
  static const bool value = is_fp_or_dword_type<T>::value;
};

/// Convert types into vector types
template <typename T> struct simd_type {
  using type = sycl::INTEL::gpu::simd<T, 1>;
};
template <typename T, int N>
struct simd_type<sycl::INTEL::gpu::detail::vector_type<T, N>> {
  using type = sycl::INTEL::gpu::simd<T, N>;
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

} // namespace detail
} // namespace gpu
} // namespace INTEL
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
