//==----------- generic_type_traits - SYCL type traits ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>             // for decorated, address_space
#include <sycl/aliases.hpp>                   // for half, cl_char, cl_double
#include <sycl/detail/helpers.hpp>            // for marray
#include <sycl/detail/type_traits.hpp>        // for is_gen_based_on_type_s...
#include <sycl/half_type.hpp>                 // for BIsRepresentationT
#include <sycl/multi_ptr.hpp>                 // for multi_ptr, address_spa...

#include <sycl/ext/oneapi/bfloat16.hpp> // for bfloat16 storage type.

#include <cstddef>     // for byte
#include <cstdint>     // for uint8_t
#include <limits>      // for numeric_limits
#include <type_traits> // for enable_if_t, condition...

namespace sycl {
inline namespace _V1 {
namespace detail {
template <typename T>
using is_byte = typename
#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
    std::is_same<T, std::byte>;
#else
    std::false_type;
#endif

template <typename T> inline constexpr bool is_byte_v = is_byte<T>::value;

template <typename T>
inline constexpr bool is_svgenfloatf_v =
    std::is_same_v<T, float> ||
    (is_vec_v<T> && std::is_same_v<element_type_t<T>, float>);

template <typename T>
inline constexpr bool is_svgenfloath_v =
    std::is_same_v<T, half> ||
    (is_vec_v<T> && std::is_same_v<element_type_t<T>, half>);

template <typename T>
inline constexpr bool is_sgenfloat_v =
    check_type_in_v<T, float, double, half, ext::oneapi::bfloat16>;

template <typename T>
inline constexpr bool is_vgenfloat_v =
    is_vec_v<T> && is_sgenfloat_v<element_type_t<T>>;

template <typename T>
inline constexpr bool is_genfloat_v =
    is_sgenfloat_v<T> || is_vgenfloat_v<T> ||
    (is_marray_v<T> && is_sgenfloat_v<element_type_t<T>> &&
     is_allowed_vec_size_v<num_elements_v<T>>);

template <typename T>
inline constexpr bool is_sigeninteger_v =
    check_type_in_v<T, signed char, short, int, long, long long> ||
    (std::is_same_v<T, char> && std::is_signed_v<char>);

template <typename T>
inline constexpr bool is_sugeninteger_v =
    check_type_in_v<T, unsigned char, unsigned short, unsigned int,
                    unsigned long, unsigned long long> ||
    (std::is_same_v<T, char> && std::is_unsigned_v<char>) || is_byte_v<T>;

template <typename T>
inline constexpr bool is_sgeninteger_v =
    is_sigeninteger_v<T> || is_sugeninteger_v<T>;

template <typename T>
inline constexpr bool is_geninteger_v =
    is_sgeninteger_v<T> ||
    (is_vec_v<T> && is_sgeninteger_v<element_type_t<T>>) ||
    (is_marray_v<T> && is_sgeninteger_v<element_type_t<T>> &&
     is_allowed_vec_size_v<num_elements_v<T>>);

template <typename T>
inline constexpr bool is_genbool_v =
    std::is_same_v<T, bool> ||
    (is_marray_v<T> && std::is_same_v<element_type_t<T>, bool> &&
     is_allowed_vec_size_v<num_elements_v<T>>);

template <int Size>
using fixed_width_unsigned = std::conditional_t<
    Size == 1, uint8_t,
    std::conditional_t<
        Size == 2, uint16_t,
        std::conditional_t<Size == 4, uint32_t, uint64_t>>>;

template <int Size>
using fixed_width_signed = std::conditional_t<
    Size == 1, int8_t,
    std::conditional_t<
        Size == 2, int16_t,
        std::conditional_t<Size == 4, int32_t, int64_t>>>;

// Use SFINAE so that std::complex specialization could be implemented in
// include/sycl/stl_wrappers/complex that would only be available if STL's
// <complex> is included by users. Note that "function template partial
// specialization" is not allowed, so we cannot perform that trick on
// convertToOpenCLType function directly.
template <typename T, typename = void> struct select_cl_scalar_complex_or_T {
  using type = T;
};

template <typename T>
using select_cl_scalar_complex_or_T_t =
    typename select_cl_scalar_complex_or_T<T>::type;

template <typename T> auto convertToOpenCLType(T &&x) {
  using no_ref = std::remove_reference_t<T>;
  if constexpr (is_multi_ptr_v<no_ref>) {
    return convertToOpenCLType(x.get_decorated());
  } else if constexpr (std::is_pointer_v<no_ref>) {
    // TODO: Below ignores volatile, but we didn't have a need for it yet.
    using elem_type = remove_decoration_t<std::remove_pointer_t<no_ref>>;
    using converted_elem_type_no_cv = decltype(convertToOpenCLType(
        std::declval<std::remove_const_t<elem_type>>()));
    using converted_elem_type =
        std::conditional_t<std::is_const_v<elem_type>,
                           const converted_elem_type_no_cv,
                           converted_elem_type_no_cv>;
#ifdef __SYCL_DEVICE_ONLY__
    using result_type =
        typename DecoratedType<converted_elem_type,
                               deduce_AS<no_ref>::value>::type *;
#else
    using result_type = converted_elem_type *;
#endif
    return reinterpret_cast<result_type>(x);
  } else if constexpr (is_vec_v<no_ref>) {
    using ElemTy = typename no_ref::element_type;
    // sycl::half may convert to _Float16, and we would try to instantiate
    // vec class with _Float16 DataType, which is not expected there. As
    // such, leave vector<half, N> as-is.
    using MatchingVec =
        vec<std::conditional_t<std::is_same_v<ElemTy, half>, ElemTy,
                               decltype(convertToOpenCLType(
                                   std::declval<ElemTy>()))>,
            no_ref::size()>;
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::bit_cast<typename MatchingVec::vector_t>(x);
#else
    return x.template as<MatchingVec>();
#endif
#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
  } else if constexpr (std::is_same_v<no_ref, std::byte>) {
    return static_cast<uint8_t>(x);
#endif
  } else if constexpr (std::is_integral_v<no_ref>) {
    using OpenCLType = std::conditional_t<std::is_signed_v<no_ref>,
                                          fixed_width_signed<sizeof(no_ref)>,
                                          fixed_width_unsigned<sizeof(no_ref)>>;
    static_assert(sizeof(OpenCLType) == sizeof(T));
    return static_cast<OpenCLType>(x);
  } else if constexpr (std::is_same_v<no_ref, half>) {
    using OpenCLType = sycl::detail::half_impl::BIsRepresentationT;
    static_assert(sizeof(OpenCLType) == sizeof(T));
    return static_cast<OpenCLType>(x);
  } else if constexpr (std::is_same_v<no_ref, ext::oneapi::bfloat16>) {
    // On host, don't interpret BF16 as uint16.
#ifdef __SYCL_DEVICE_ONLY__
    using OpenCLType = sycl::ext::oneapi::detail::Bfloat16StorageT;
    return sycl::bit_cast<OpenCLType>(x);
#else
    return std::forward<T>(x);
#endif
  } else if constexpr (std::is_floating_point_v<no_ref>) {
    static_assert(std::is_same_v<no_ref, float> ||
                      std::is_same_v<no_ref, double>,
                  "Other FP types are not expected/supported (yet?)");
    static_assert(std::is_same_v<float, sycl::opencl::cl_float> &&
                  std::is_same_v<double, sycl::opencl::cl_double>);
    return std::forward<T>(x);
  } else {
    using OpenCLType = select_cl_scalar_complex_or_T_t<no_ref>;
    static_assert(sizeof(OpenCLType) == sizeof(T));
    return static_cast<OpenCLType>(x);
  }
}

template <typename T>
using ConvertToOpenCLType_t = decltype(convertToOpenCLType(std::declval<T>()));

template <typename To, typename From> auto convertFromOpenCLTypeFor(From &&x) {
  if constexpr (std::is_same_v<To, bool> &&
                std::is_same_v<std::remove_reference_t<From>, bool>) {
    // FIXME: Something seems to be wrong elsewhere...
    return x;
  } else {
    using OpenCLType = decltype(convertToOpenCLType(std::declval<To>()));
    static_assert(std::is_same_v<std::remove_reference_t<From>, OpenCLType>);
    static_assert(sizeof(OpenCLType) == sizeof(To));
    using To_noref = std::remove_reference_t<To>;
    using From_noref = std::remove_reference_t<From>;
    if constexpr (is_vec_v<To_noref> && is_vec_v<From_noref>)
      return x.template as<To_noref>();
    else if constexpr (is_vec_v<To_noref> && is_ext_vector_v<From_noref>)
      return To_noref{bit_cast<typename To_noref::vector_t>(x)};
    else
      return static_cast<To>(x);
  }
}

// Used for all, any and select relational built-in functions
template <typename T> inline constexpr T msbMask(T) {
  using UT = make_unsigned_t<T>;
  return T(UT(1) << (sizeof(T) * 8 - 1));
}

template <typename T> inline constexpr bool msbIsSet(const T x) {
  return (x & msbMask(x));
}

template <typename T> static constexpr T max_v() {
  return (std::numeric_limits<T>::max)();
}

template <typename T> static constexpr T min_v() {
  return (std::numeric_limits<T>::min)();
}
} // namespace detail
} // namespace _V1
} // namespace sycl
