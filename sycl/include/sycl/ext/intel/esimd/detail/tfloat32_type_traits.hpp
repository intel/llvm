//==-------------- tfloat32_type_traits.hpp - DPC++ Explicit SIMD API
//----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implementation of SIMD element type traits for the tfloat32 type.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/esimd/detail/elem_type_traits.hpp>
#include <sycl/ext/intel/esimd/detail/intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/tfloat32.hpp>

/// @cond ESIMD_DETAIL

namespace sycl {
inline namespace _V1 {
namespace ext::intel::esimd::detail {

// Standalone definitions to use w/o instantiating element_type_traits.
using tfloat32 = sycl::ext::intel::experimental::esimd::tfloat32;

template <> struct element_type_traits<tfloat32> {
  using RawT = unsigned int;
  using EnclosingCppT = float;

  static constexpr bool use_native_cpp_ops = false;
  static constexpr bool is_floating_point = true;
};

// ------------------- Type conversion traits

template <int N> struct vector_conversion_traits<tfloat32, N> {
  using StdT = __cpp_t<tfloat32>;
  using RawT = __raw_t<tfloat32>;

  static ESIMD_INLINE vector_type_t<RawT, N>
  convert_to_raw(vector_type_t<StdT, N> Val) {
#ifdef __SYCL_DEVICE_ONLY__
    vector_type_t<RawT, N> Result = __esimd_tf32_cvt<RawT, StdT, N>(Val);
    return Result;
#else
    vector_type_t<RawT, N> Output = 0;

    for (int i = 0; i < N; i++) {
      Output[i] = sycl::bit_cast<RawT>(static_cast<tfloat32>(Val[i]));
    }
    return Output;
#endif
  }

  static ESIMD_INLINE vector_type_t<StdT, N>
  convert_to_cpp(vector_type_t<RawT, N> Val) {
    vector_type_t<StdT, N> Result = sycl::bit_cast<vector_type_t<StdT, N>>(Val);
    return Result;
  }
};

template <> struct scalar_conversion_traits<tfloat32> {
  using RawT = __raw_t<tfloat32>;

  static ESIMD_INLINE RawT bitcast_to_raw(tfloat32 Val) {
    return sycl::bit_cast<RawT>(Val);
  }

  static ESIMD_INLINE tfloat32 bitcast_to_wrapper(RawT Val) {
    return sycl::bit_cast<tfloat32>(Val);
  }
};

// Misc
inline std::ostream &operator<<(std::ostream &O, tfloat32 const &rhs) {
  O << static_cast<float>(rhs);
  return O;
}

template <> struct is_esimd_arithmetic_type<tfloat32, void> : std::true_type {};

} // namespace ext::intel::esimd::detail
} // namespace _V1
} // namespace sycl

/// @endcond ESIMD_DETAIL
