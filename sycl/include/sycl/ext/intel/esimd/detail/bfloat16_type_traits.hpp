//==-------------- bfloat16_type_traits.hpp - DPC++ Explicit SIMD API ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implementation of SIMD element type traits for the bfloat16 type.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/esimd/detail/elem_type_traits.hpp>
#include <sycl/ext/intel/esimd/detail/intrin.hpp>

#include <sycl/ext/oneapi/bfloat16.hpp>

/// @cond ESIMD_DETAIL

namespace sycl {
inline namespace _V1 {
namespace ext::intel::esimd::detail {

using bfloat16 = sycl::ext::oneapi::bfloat16;

template <> struct element_type_traits<bfloat16> {
  // TODO map the raw type to __bf16 once SPIRV target supports it:
  using RawT = uint_type_t<sizeof(bfloat16)>;
  // Nearest standard enclosing C++ type to delegate natively unsupported
  // operations to:
  using EnclosingCppT = float;
  // Can't map bfloat16 operations to opertations on RawT:
  static constexpr bool use_native_cpp_ops = false;
  static constexpr bool is_floating_point = true;
};

#ifdef __SYCL_DEVICE_ONLY__
// VC BE-specific glitch
// @llvm.genx.bf.cvt uses half (_Float16) as bit representation for bfloat16
using vc_be_bfloat16_raw_t = _Float16;
#endif // __SYCL_DEVICE_ONLY__

// ------------------- Type conversion traits

template <int N> struct vector_conversion_traits<bfloat16, N> {
  using StdT = __cpp_t<bfloat16>;
  using StdVecT = vector_type_t<StdT, N>;
  using RawT = __raw_t<bfloat16>;

  static ESIMD_INLINE vector_type_t<RawT, N>
  convert_to_raw(vector_type_t<StdT, N> Val) {
#ifdef __SYCL_DEVICE_ONLY__
    using RawVecT = vector_type_t<vc_be_bfloat16_raw_t, N>;
    RawVecT ConvVal = __esimd_bf_cvt<vc_be_bfloat16_raw_t, StdT, N>(Val);
    // cast from _Float16 to int16_t:
    return sycl::bit_cast<vector_type_t<RawT, N>>(ConvVal);
#else
    __ESIMD_UNSUPPORTED_ON_HOST;
#endif // __SYCL_DEVICE_ONLY__
  }

  static ESIMD_INLINE vector_type_t<StdT, N>
  convert_to_cpp(vector_type_t<RawT, N> Val) {
#ifdef __SYCL_DEVICE_ONLY__
    using RawVecT = vector_type_t<vc_be_bfloat16_raw_t, N>;
    RawVecT Bits = sycl::bit_cast<RawVecT>(Val);
    return __esimd_bf_cvt<StdT, vc_be_bfloat16_raw_t, N>(Bits);
#else
    __ESIMD_UNSUPPORTED_ON_HOST;
#endif // __SYCL_DEVICE_ONLY__
  }
};

// TODO: remove bitcasts from the scalar_conversion_traits, and replace with
// sycl::bit_cast directly
template <> struct scalar_conversion_traits<bfloat16> {
  using RawT = __raw_t<bfloat16>;

  static ESIMD_INLINE RawT bitcast_to_raw(bfloat16 Val) {
    return sycl::bit_cast<RawT>(Val);
  }

  static ESIMD_INLINE bfloat16 bitcast_to_wrapper(RawT Val) {
    return sycl::bit_cast<bfloat16>(Val);
  }
};

// bfloat16 uses default inefficient implementations of std C++ operations,
// hence no specializations of other traits.

// Misc
inline std::ostream &operator<<(std::ostream &O, bfloat16 const &rhs) {
  O << static_cast<float>(rhs);
  return O;
}

template <> struct is_esimd_arithmetic_type<bfloat16, void> : std::true_type {};

} // namespace ext::intel::esimd::detail
} // namespace _V1
} // namespace sycl

/// @endcond ESIMD_DETAIL
