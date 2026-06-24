//==-------------- fp4_type_traits.hpp - DPC++ Explicit SIMD API ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implementation of SIMD element type traits for the fp4 type.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/bit_cast.hpp>
#include <sycl/ext/intel/esimd/detail/elem_type_traits.hpp>
#include <sycl/ext/intel/experimental/esimd/fp4.hpp>

/// @cond ESIMD_DETAIL

namespace sycl {
inline namespace _V1 {
namespace ext::intel::esimd::detail {

// Standalone definitions to use w/o instantiating element_type_traits.
using fp4_S1E2M1 = sycl::ext::intel::experimental::esimd::fp4_S1E2M1;

// ------------------- Type conversion traits

template <> struct element_type_traits<fp4_S1E2M1> {
  using RawT = unsigned char;
  using EnclosingCppT = unsigned char;

  static inline constexpr bool use_native_cpp_ops = false;
  static inline constexpr bool is_floating_point = true;
};

// ------------------- Type conversion traits

template <int N> struct vector_conversion_traits<fp4_S1E2M1, N> {
  using StdT = __cpp_t<fp4_S1E2M1>;
  using RawT = __raw_t<fp4_S1E2M1>;

  static ESIMD_INLINE vector_type_t<RawT, N>
  convert_to_raw(vector_type_t<StdT, N> Val) {
    return __ESIMD_DNS::convert_vector<RawT, StdT, N>(Val);
  }

  static ESIMD_INLINE vector_type_t<StdT, N>
  convert_to_cpp(vector_type_t<RawT, N> Val) {
    return __ESIMD_DNS::convert_vector<StdT, RawT, N>(Val);
  }
};

template <> struct scalar_conversion_traits<fp4_S1E2M1> {
  using RawT = __raw_t<fp4_S1E2M1>;

  static ESIMD_INLINE RawT bitcast_to_raw(fp4_S1E2M1 Val) {
    return sycl::bit_cast<RawT>(Val);
  }

  static ESIMD_INLINE fp4_S1E2M1 bitcast_to_wrapper(RawT Val) {
    return sycl::bit_cast<fp4_S1E2M1>(Val);
  }
};

} // namespace ext::intel::esimd::detail
} // namespace _V1
} // namespace sycl

template <typename T>
inline std::enable_if_t<
    std::is_same_v<T, sycl::ext::intel::experimental::esimd::fp4_S1E2M1>,
    std::ostream &>
operator<<(std::ostream &O, T const &rhs) {
  O << rhs.raw();
  return O;
}

/// @endcond ESIMD_DETAIL
