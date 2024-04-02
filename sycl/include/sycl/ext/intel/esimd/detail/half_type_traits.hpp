//==-------------- half_type_traits.hpp - DPC++ Explicit SIMD API ----------==//
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

#include <sycl/half_type.hpp>

/// @cond ESIMD_DETAIL

namespace sycl {
inline namespace _V1 {
namespace ext::intel::esimd::detail {

// Standalone definitions to use w/o instantiating element_type_traits.
#ifdef __SYCL_DEVICE_ONLY__
// Can't use sycl::detail::half_impl::StorageT as RawT for both host and
// device as it still maps to struct on/ host (even though the struct is a
// trivial wrapper around uint16_t), and for ESIMD we need a type which can be
// an element of clang vector.
using half_raw_type = sycl::detail::half_impl::StorageT;
// On device, _Float16 is native Cpp type, so it is the enclosing C++ type
using half_enclosing_cpp_type = half_raw_type;
#else
using half_raw_type = uint16_t;
using half_enclosing_cpp_type = float;
#endif // __SYCL_DEVICE_ONLY__

template <> struct element_type_traits<sycl::half> {
  using RawT = half_raw_type;
  using EnclosingCppT = half_enclosing_cpp_type;
#ifdef __SYCL_DEVICE_ONLY__
  // On device, operations on half are translated to operations on _Float16,
  // which is natively supported by the device compiler
  static constexpr bool use_native_cpp_ops = true;
#else
  // On host, we can't use native Cpp '+', '-' etc. over uint16_t to emulate the
  // operations on half type.
  static constexpr bool use_native_cpp_ops = false;
#endif // __SYCL_DEVICE_ONLY__

  static constexpr bool is_floating_point = true;
};

// ------------------- Type conversion traits

template <int N> struct vector_conversion_traits<sycl::half, N> {
  using StdT = half_enclosing_cpp_type;
  using RawT = half_raw_type;

  static ESIMD_INLINE vector_type_t<RawT, N>
  convert_to_raw(vector_type_t<StdT, N> Val)
#ifdef __SYCL_DEVICE_ONLY__
      // use_native_cpp_ops trait is true, so must not be implemented
      ;
#else
  {
    __ESIMD_UNSUPPORTED_ON_HOST;
  }
#endif // __SYCL_DEVICE_ONLY__

  static ESIMD_INLINE vector_type_t<StdT, N>
  convert_to_cpp(vector_type_t<RawT, N> Val)
#ifdef __SYCL_DEVICE_ONLY__
      // use_native_cpp_ops trait is true, so must not be implemented
      ;
#else
  {
    __ESIMD_UNSUPPORTED_ON_HOST;
  }
#endif // __SYCL_DEVICE_ONLY__
};

// Proxy class to access bit representation of a wrapper type both on host and
// device. Declared as friend to the sycl::half.
// TODO add this functionality to sycl type implementation? With C++20,
// std::bit_cast should be a good replacement.
class WrapperElementTypeProxy {
public:
  static ESIMD_INLINE half_raw_type bitcast_to_raw_scalar(sycl::half Val) {
#ifdef __SYCL_DEVICE_ONLY__
    return Val.Data;
#else
    return Val.Data.Buf;
#endif // __SYCL_DEVICE_ONLY__
  }

  static ESIMD_INLINE sycl::half bitcast_to_wrapper_scalar(half_raw_type Val) {
#ifndef __SYCL_DEVICE_ONLY__
    __ESIMD_UNSUPPORTED_ON_HOST;
#else
    sycl::half Res;
    Res.Data = Val;
    return Res;
#endif // __SYCL_DEVICE_ONLY__
  }
};

template <> struct scalar_conversion_traits<sycl::half> {
  using RawT = half_raw_type;

  static ESIMD_INLINE RawT bitcast_to_raw(sycl::half Val) {
    return WrapperElementTypeProxy::bitcast_to_raw_scalar(Val);
  }

  static ESIMD_INLINE sycl::half bitcast_to_wrapper(RawT Val) {
    return WrapperElementTypeProxy::bitcast_to_wrapper_scalar(Val);
  }
};

#ifdef __SYCL_DEVICE_ONLY__
template <>
struct is_esimd_arithmetic_type<half_raw_type, void> : std::true_type {};
#endif // __SYCL_DEVICE_ONLY__

template <>
struct is_esimd_arithmetic_type<sycl::half, void> : std::true_type {};

// Misc
inline std::ostream &operator<<(std::ostream &O, sycl::half const &rhs) {
  O << static_cast<float>(rhs);
  return O;
}

inline std::istream &operator>>(std::istream &I, sycl::half &rhs) {
  float ValFloat = 0.0f;
  I >> ValFloat;
  rhs = ValFloat;
  return I;
}

} // namespace ext::intel::esimd::detail
} // namespace _V1
} // namespace sycl

/// @endcond ESIMD_DETAIL
