//==-------------- tfloat32_type_traits.hpp - DPC++ Explicit SIMD API
//----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implementation of SIMD element type traits for the sycl::half type.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/esimd/detail/elem_type_traits.hpp>
#include <sycl/ext/intel/esimd/detail/intrin.hpp>
#include <sycl/ext/oneapi/experimental/tfloat32.hpp>

/// @cond ESIMD_DETAIL

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::intel::esimd::detail {

// Standalone definitions to use w/o instantiating element_type_traits.
using tfloat32 = sycl::ext::oneapi::experimental::tfloat32;
using tfloat32_raw_type = float;
using tfloat32_enclosing_cpp_type = float;

template <> struct element_type_traits<tfloat32> {
  using RawT = tfloat32_raw_type;
  using EnclosingCppT = tfloat32_enclosing_cpp_type;

  static inline constexpr bool use_native_cpp_ops = false;
  static inline constexpr bool is_floating_point = true;
};

// ------------------- Type conversion traits

template <int N> struct vector_conversion_traits<tfloat32, N> {
  using StdT = tfloat32_enclosing_cpp_type;
  using RawT = tfloat32_raw_type;

  static ESIMD_INLINE vector_type_t<RawT, N>
  convert_to_raw(vector_type_t<StdT, N> Val) {
#ifdef __SYCL_DEVICE_ONLY__
    using RawVecT = vector_type_t<float, N>;
    RawVecT ConvVal = __esimd_tf32_cvt<float, StdT, N>(Val);
    return ConvVal;
#else
    vector_type_t<tfloat32, N> Output = 0;

    for (int i = 0; i < N; i += 1) {
      Output[i] = tfloat32::from_float(Val[i]);
    }
    return Output;
#endif // __SYCL_DEVICE_ONLY__
  }

  static ESIMD_INLINE vector_type_t<StdT, N>
  convert_to_cpp(vector_type_t<RawT, N> Val) {
    return Val;
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

inline std::istream &operator>>(std::istream &I, tfloat32 &rhs) {
  float ValFloat = 0.0f;
  I >> ValFloat;
  rhs = ValFloat;
  return I;
}

} // namespace ext::intel::esimd::detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

/// @endcond ESIMD_DETAIL
