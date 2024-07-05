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
#include <sycl/detail/generic_type_lists.hpp> // for nonconst_address_space...
#include <sycl/detail/helpers.hpp>            // for marray
#include <sycl/detail/type_list.hpp>          // for is_contained, find_sam...
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
template <int N> struct Boolean;

template <typename T>
inline constexpr bool is_svgenfloatf_v =
    is_contained_v<T, gtl::scalar_vector_float_list>;

template <typename T>
inline constexpr bool is_svgenfloatd_v =
    is_contained_v<T, gtl::scalar_vector_double_list>;

template <typename T>
inline constexpr bool is_half_v = is_contained_v<T, gtl::scalar_half_list>;

template <typename T>
inline constexpr bool is_bfloat16_v =
    is_contained_v<T, gtl::scalar_bfloat16_list>;

template <typename T>
inline constexpr bool is_half_or_bf16_v =
    is_contained_v<T, gtl::half_bfloat16_list>;

template <typename T>
inline constexpr bool is_svgenfloath_v =
    is_contained_v<T, gtl::scalar_vector_half_list>;

template <typename T>
inline constexpr bool is_genfloat_v = is_contained_v<T, gtl::floating_list>;

template <typename T>
inline constexpr bool is_sgenfloat_v =
    is_contained_v<T, gtl::scalar_floating_list>;

template <typename T>
inline constexpr bool is_vgenfloat_v =
    is_contained_v<T, gtl::vector_floating_list>;

template <typename T>
inline constexpr bool is_svgenfloat_v =
    is_contained_v<T, gtl::scalar_vector_floating_list>;

template <typename T>
inline constexpr bool is_mgenfloat_v =
    is_marray_v<T> && is_svgenfloat_v<get_elem_type_t<T>>;

template <typename T>
inline constexpr bool is_gengeofloat_v = is_contained_v<T, gtl::geo_float_list>;

template <typename T>
inline constexpr bool is_gengeodouble_v =
    is_contained_v<T, gtl::geo_double_list>;

template <typename T>
inline constexpr bool is_gengeomarrayfloat_v =
    is_contained_v<T, gtl::marray_geo_float_list>;

template <typename T>
inline constexpr bool is_gengeomarray_v =
    is_contained_v<T, gtl::marray_geo_list>;

template <typename T>
inline constexpr bool is_gengeohalf_v = is_contained_v<T, gtl::geo_half_list>;

template <typename T>
inline constexpr bool is_vgengeofloat_v =
    is_contained_v<T, gtl::vector_geo_float_list>;

template <typename T>
inline constexpr bool is_vgengeodouble_v =
    is_contained_v<T, gtl::vector_geo_double_list>;

template <typename T>
inline constexpr bool is_vgengeohalf_v =
    is_contained_v<T, gtl::vector_geo_half_list>;

template <typename T>
inline constexpr bool is_sgengeo_v = is_contained_v<T, gtl::scalar_geo_list>;

template <typename T>
inline constexpr bool is_vgengeo_v = is_contained_v<T, gtl::vector_geo_list>;

template <typename T>
inline constexpr bool is_gencrossfloat_v =
    is_contained_v<T, gtl::cross_float_list>;

template <typename T>
inline constexpr bool is_gencrossdouble_v =
    is_contained_v<T, gtl::cross_double_list>;

template <typename T>
inline constexpr bool is_gencrosshalf_v =
    is_contained_v<T, gtl::cross_half_list>;

template <typename T>
inline constexpr bool is_gencross_v =
    is_contained_v<T, gtl::cross_floating_list>;

template <typename T>
inline constexpr bool is_gencrossmarray_v =
    is_contained_v<T, gtl::cross_marray_list>;

template <typename T>
inline constexpr bool is_ugenint_v = is_contained_v<T, gtl::unsigned_int_list>;

template <typename T>
inline constexpr bool is_intn_v =
    is_contained_v<T, gtl::vector_signed_int_list>;

template <typename T>
inline constexpr bool is_genint_v = is_contained_v<T, gtl::signed_int_list>;

template <typename T>
inline constexpr bool is_geninteger_v = is_contained_v<T, gtl::integer_list>;

template <typename T>
using is_geninteger = std::bool_constant<is_geninteger_v<T>>;

template <typename T>
inline constexpr bool is_igeninteger_v =
    is_contained_v<T, gtl::signed_integer_list>;

template <typename T>
using is_igeninteger = std::bool_constant<is_igeninteger_v<T>>;

template <typename T>
inline constexpr bool is_ugeninteger_v =
    is_contained_v<T, gtl::unsigned_integer_list>;

template <typename T>
using is_ugeninteger = std::bool_constant<is_ugeninteger_v<T>>;

template <typename T>
inline constexpr bool is_sgeninteger_v =
    is_contained_v<T, gtl::scalar_integer_list>;

template <typename T>
inline constexpr bool is_vgeninteger_v =
    is_contained_v<T, gtl::vector_integer_list>;

template <typename T>
inline constexpr bool is_sigeninteger_v =
    is_contained_v<T, gtl::scalar_signed_integer_list>;

template <typename T>
inline constexpr bool is_sugeninteger_v =
    is_contained_v<T, gtl::scalar_unsigned_integer_list>;

template <typename T>
inline constexpr bool is_vigeninteger_v =
    is_contained_v<T, gtl::vector_signed_integer_list>;

template <typename T>
inline constexpr bool is_vugeninteger_v =
    is_contained_v<T, gtl::vector_unsigned_integer_list>;

template <typename T>
inline constexpr bool is_genbool_v = is_contained_v<T, gtl::bool_list>;

template <typename T>
inline constexpr bool is_gentype_v = is_contained_v<T, gtl::basic_list>;

template <typename T>
inline constexpr bool is_vgentype_v = is_contained_v<T, gtl::vector_basic_list>;

template <typename T>
inline constexpr bool is_sgentype_v = is_contained_v<T, gtl::scalar_basic_list>;

template <typename T>
inline constexpr bool is_igeninteger8bit_v =
    is_gen_based_on_type_sizeof_v<T, 1, is_igeninteger>;

template <typename T>
inline constexpr bool is_igeninteger16bit_v =
    is_gen_based_on_type_sizeof_v<T, 2, is_igeninteger>;

template <typename T>
inline constexpr bool is_igeninteger32bit_v =
    is_gen_based_on_type_sizeof_v<T, 4, is_igeninteger>;

template <typename T>
inline constexpr bool is_igeninteger64bit_v =
    is_gen_based_on_type_sizeof_v<T, 8, is_igeninteger>;

template <typename T>
inline constexpr bool is_ugeninteger8bit_v =
    is_gen_based_on_type_sizeof_v<T, 1, is_ugeninteger>;

template <typename T>
inline constexpr bool is_ugeninteger16bit_v =
    is_gen_based_on_type_sizeof_v<T, 2, is_ugeninteger>;

template <typename T>
inline constexpr bool is_ugeninteger32bit_v =
    is_gen_based_on_type_sizeof_v<T, 4, is_ugeninteger>;

template <typename T>
inline constexpr bool is_ugeninteger64bit_v =
    is_gen_based_on_type_sizeof_v<T, 8, is_ugeninteger>;

template <typename T>
inline constexpr bool is_genintptr_v =
    is_pointer_v<T> && is_genint_v<remove_pointer_t<T>> &&
    is_address_space_compliant_v<T, gvl::nonconst_address_space_list>;

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline constexpr bool is_genintptr_marray_v =
    std::is_same_v<T, sycl::marray<marray_element_t<T>, T::size()>> &&
    is_genint_v<marray_element_t<remove_pointer_t<T>>> &&
    is_address_space_compliant_v<multi_ptr<T, AddressSpace, IsDecorated>,
                                 gvl::nonconst_address_space_list> &&
    (IsDecorated == access::decorated::yes ||
     IsDecorated == access::decorated::no);

template <typename T>
inline constexpr bool is_genfloatptr_v =
    is_pointer_v<T> && is_genfloat_v<remove_pointer_t<T>> &&
    is_address_space_compliant_v<T, gvl::nonconst_address_space_list>;

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
inline constexpr bool is_genfloatptr_marray_v =
    is_mgenfloat_v<T> &&
    is_address_space_compliant_v<multi_ptr<T, AddressSpace, IsDecorated>,
                                 gvl::nonconst_address_space_list> &&
    (IsDecorated == access::decorated::yes ||
     IsDecorated == access::decorated::no);

template <typename T>
using is_byte = typename
#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
    std::is_same<T, std::byte>;
#else
    std::false_type;
#endif

template <typename T> inline constexpr bool is_byte_v = is_byte<T>::value;

template <typename T>
using make_floating_point_t = make_type_t<T, gtl::scalar_floating_list>;

template <typename T>
using make_singed_integer_t = make_type_t<T, gtl::scalar_signed_integer_list>;

template <typename T>
using make_unsinged_integer_t =
    make_type_t<T, gtl::scalar_unsigned_integer_list>;

template <int Size>
using cl_unsigned = std::conditional_t<
    Size == 1, opencl::cl_uchar,
    std::conditional_t<
        Size == 2, opencl::cl_ushort,
        std::conditional_t<Size == 4, opencl::cl_uint, opencl::cl_ulong>>>;

// select_apply_cl_scalar_t selects from T8/T16/T32/T64 basing on
// sizeof(IN).  expected to handle scalar types.
template <typename T, typename T8, typename T16, typename T32, typename T64>
using select_apply_cl_scalar_t = std::conditional_t<
    sizeof(T) == 1, T8,
    std::conditional_t<sizeof(T) == 2, T16,
                       std::conditional_t<sizeof(T) == 4, T32, T64>>>;

// Shortcuts for selecting scalar int/unsigned int/fp type.
template <typename T>
using select_cl_scalar_integral_signed_t =
    select_apply_cl_scalar_t<T, sycl::opencl::cl_char, sycl::opencl::cl_short,
                             sycl::opencl::cl_int, sycl::opencl::cl_long>;

template <typename T>
using select_cl_scalar_integral_unsigned_t =
    select_apply_cl_scalar_t<T, sycl::opencl::cl_uchar, sycl::opencl::cl_ushort,
                             sycl::opencl::cl_uint, sycl::opencl::cl_ulong>;

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
    using MatchingVec = vec<std::conditional_t<is_half_v<ElemTy>, ElemTy,
                                               decltype(convertToOpenCLType(
                                                   std::declval<ElemTy>()))>,
                            no_ref::size()>;
#ifdef __SYCL_DEVICE_ONLY__

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
    // TODO: for some mysterious reasons on NonUniformGroups E2E tests fail if
    // we use the "else" version only. I suspect that's an issues with
    // non-uniform groups implementation.
    if constexpr (std::is_same_v<MatchingVec, no_ref>)
      return static_cast<typename MatchingVec::vector_t>(x);
    else
      return static_cast<typename MatchingVec::vector_t>(
          x.template as<MatchingVec>());
#else  // __INTEL_PREVIEW_BREAKING_CHANGES
    return sycl::bit_cast<typename MatchingVec::vector_t>(x);
#endif // __INTEL_PREVIEW_BREAKING_CHANGES

#else
    return x.template as<MatchingVec>();
#endif
  } else if constexpr (is_boolean_v<no_ref>) {
#ifdef __SYCL_DEVICE_ONLY__
    if constexpr (std::is_same_v<Boolean<1>, no_ref>) {
      // Or should it be "int"?
      return std::forward<T>(x);
    } else {
      return static_cast<typename no_ref::vector_t>(x);
    }
#else
    return std::forward<T>(x);
#endif
#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
  } else if constexpr (std::is_same_v<no_ref, std::byte>) {
    return static_cast<uint8_t>(x);
#endif
  } else if constexpr (std::is_integral_v<no_ref>) {
    using OpenCLType =
        std::conditional_t<std::is_signed_v<no_ref>,
                           select_cl_scalar_integral_signed_t<no_ref>,
                           select_cl_scalar_integral_unsigned_t<no_ref>>;
    static_assert(sizeof(OpenCLType) == sizeof(T));
    return static_cast<OpenCLType>(x);
  } else if constexpr (is_half_v<no_ref>) {
    using OpenCLType = sycl::detail::half_impl::BIsRepresentationT;
    static_assert(sizeof(OpenCLType) == sizeof(T));
    return static_cast<OpenCLType>(x);
  } else if constexpr (is_bfloat16_v<no_ref>) {
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

// Try to get vector element count or 1 otherwise
template <typename T> struct GetNumElements {
  static constexpr int value = 1;
};
template <typename Type, int NumElements>
struct GetNumElements<typename sycl::vec<Type, NumElements>> {
  static constexpr int value = NumElements;
};
template <int N> struct GetNumElements<typename sycl::detail::Boolean<N>> {
  static constexpr int value = N;
};

// TryToGetElementType<T>::type is T::element_type or T::value_type if those
// exist, otherwise T.
template <typename T> class TryToGetElementType {
  static T check(...);
  template <typename A> static typename A::element_type check(const A &);
  template <typename A, typename = std::enable_if_t<!std::is_same_v<
                            typename A::element_type, typename A::value_type>>>
  static typename A::value_type check(const A &);

public:
  using type = decltype(check(T()));
  static constexpr bool value = !std::is_same_v<T, type>;
};

template <typename T> static constexpr T max_v() {
  return (std::numeric_limits<T>::max)();
}

template <typename T> static constexpr T min_v() {
  return (std::numeric_limits<T>::min)();
}
} // namespace detail
} // namespace _V1
} // namespace sycl
