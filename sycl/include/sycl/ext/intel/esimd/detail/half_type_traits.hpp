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
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::intel::esimd::detail {

template <class T>
struct element_type_traits<T, std::enable_if_t<std::is_same_v<T, sycl::half>>> {
  // Can't use sycl::detail::half_impl::StorageT as RawT for both host and
  // device as it still maps to struct on/ host (even though the struct is a
  // trivial wrapper around uint16_t), and for ESIMD we need a type which can be
  // an element of clang vector.
#ifdef __SYCL_DEVICE_ONLY__
  using RawT = sycl::detail::half_impl::StorageT;
  // On device, _Float16 is native Cpp type, so it is the enclosing C++ type
  using EnclosingCppT = RawT;
  // On device, operations on half are translated to operations on _Float16,
  // which is natively supported by the device compiler
  static inline constexpr bool use_native_cpp_ops = true;
#else
  using RawT = uint16_t;
  using EnclosingCppT = float;
  // On host, we can't use native Cpp '+', '-' etc. over uint16_t to emulate the
  // operations on half type.
  static inline constexpr bool use_native_cpp_ops = false;
#endif // __SYCL_DEVICE_ONLY__

  static inline constexpr bool is_floating_point = true;
};

// ------------------- Type conversion traits

template <int N> struct vector_conversion_traits<sycl::half, N> {
  using StdT = __cpp_t<sycl::half>;
  using RawT = __raw_t<sycl::half>;

  static ESIMD_INLINE vector_type_t<RawT, N>
  convert_to_raw(vector_type_t<StdT, N> Val)
#ifdef __SYCL_DEVICE_ONLY__
      // use_native_cpp_ops trait is true, so must not be implemented
      ;
#else
  {
    vector_type_t<__raw_t<sycl::half>, N> Output = 0;

    for (int i = 0; i < N; i += 1) {
      // 1. Convert Val[i] to float (x) using c++ static_cast
      // 2. Convert x to half (using float2half)
      // 3. Output[i] = half_of(x)
      Output[i] = ::sycl::detail::float2Half(static_cast<float>(Val[i]));
    }
    return Output;
  }
#endif // __SYCL_DEVICE_ONLY__

  static ESIMD_INLINE vector_type_t<StdT, N>
  convert_to_cpp(vector_type_t<RawT, N> Val)
#ifdef __SYCL_DEVICE_ONLY__
      // use_native_cpp_ops trait is true, so must not be implemented
      ;
#else
  {
    vector_type_t<StdT, N> Output;

    for (int i = 0; i < N; i += 1) {
      // 1. Convert Val[i] to float y(using half2float)
      // 2. Convert y to StdT using c++ static_cast
      // 3. Store in Output[i]
      Output[i] = static_cast<StdT>(::sycl::detail::half2Float(Val[i]));
    }
    return Output;
  }
#endif // __SYCL_DEVICE_ONLY__
};

// WrapperElementTypeProxy (a friend of sycl::half) must be used to access
// private fields of the sycl::half.
template <>
ESIMD_INLINE __raw_t<sycl::half>
WrapperElementTypeProxy::bitcast_to_raw_scalar<sycl::half>(sycl::half Val) {
#ifdef __SYCL_DEVICE_ONLY__
  return Val.Data;
#else
  return Val.Data.Buf;
#endif // __SYCL_DEVICE_ONLY__
}

template <>
ESIMD_INLINE sycl::half
WrapperElementTypeProxy::bitcast_to_wrapper_scalar<sycl::half>(
    __raw_t<sycl::half> Val) {
#ifndef __SYCL_DEVICE_ONLY__
  return sycl::half(::sycl::detail::host_half_impl::half(Val));
#else
  sycl::half Res;
  Res.Data = Val;
  return Res;
#endif // __SYCL_DEVICE_ONLY__
}

template <> struct scalar_conversion_traits<sycl::half> {
  using RawT = __raw_t<sycl::half>;

  static ESIMD_INLINE RawT bitcast_to_raw(sycl::half Val) {
    return WrapperElementTypeProxy::bitcast_to_raw_scalar<sycl::half>(Val);
  }

  static ESIMD_INLINE sycl::half bitcast_to_wrapper(RawT Val) {
    return WrapperElementTypeProxy::bitcast_to_wrapper_scalar<sycl::half>(Val);
  }
};

#ifdef __SYCL_DEVICE_ONLY__
template <>
struct is_esimd_arithmetic_type<__raw_t<sycl::half>, void> : std::true_type {};
#endif // __SYCL_DEVICE_ONLY__

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
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

/// @endcond ESIMD_DETAIL
