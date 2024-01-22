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

#include <cstddef>     // for byte
#include <cstdint>     // for uint8_t
#include <limits>      // for numeric_limits
#include <type_traits> // for enable_if_t, condition...

namespace sycl {
inline namespace _V1 {
namespace detail {
template <int N> struct Boolean;

template <typename T>
inline constexpr bool is_floatn_v = is_contained_v<T, gtl::vector_float_list>;

template <typename T>
inline constexpr bool is_genfloatf_v = is_contained_v<T, gtl::float_list>;

template <typename T>
inline constexpr bool is_svgenfloatf_v =
    is_contained_v<T, gtl::scalar_vector_float_list>;

template <typename T>
inline constexpr bool is_doublen_v = is_contained_v<T, gtl::vector_double_list>;

template <typename T>
inline constexpr bool is_genfloatd_v = is_contained_v<T, gtl::double_list>;

template <typename T>
inline constexpr bool is_svgenfloatd_v =
    is_contained_v<T, gtl::scalar_vector_double_list>;

template <typename T>
inline constexpr bool is_halfn_v = is_contained_v<T, gtl::vector_half_list>;

template <typename T>
inline constexpr bool is_genfloath_v = is_contained_v<T, gtl::half_list>;

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
inline constexpr bool is_charn_v =
    is_contained_v<T, gtl::vector_default_char_list>;

template <typename T>
inline constexpr bool is_scharn_v =
    is_contained_v<T, gtl::vector_signed_char_list>;

template <typename T>
inline constexpr bool is_ucharn_v =
    is_contained_v<T, gtl::vector_unsigned_char_list>;

template <typename T>
inline constexpr bool is_igenchar_v = is_contained_v<T, gtl::signed_char_list>;

template <typename T>
inline constexpr bool is_ugenchar_v =
    is_contained_v<T, gtl::unsigned_char_list>;

template <typename T>
inline constexpr bool is_genchar_v = is_contained_v<T, gtl::char_list>;

template <typename T>
inline constexpr bool is_shortn_v =
    is_contained_v<T, gtl::vector_signed_short_list>;

template <typename T>
inline constexpr bool is_genshort_v = is_contained_v<T, gtl::signed_short_list>;

template <typename T>
inline constexpr bool is_ushortn_v =
    is_contained_v<T, gtl::vector_unsigned_short_list>;

template <typename T>
inline constexpr bool is_ugenshort_v =
    is_contained_v<T, gtl::unsigned_short_list>;

template <typename T>
inline constexpr bool is_uintn_v =
    is_contained_v<T, gtl::vector_unsigned_int_list>;

template <typename T>
inline constexpr bool is_ugenint_v = is_contained_v<T, gtl::unsigned_int_list>;

template <typename T>
inline constexpr bool is_intn_v =
    is_contained_v<T, gtl::vector_signed_int_list>;

template <typename T>
inline constexpr bool is_genint_v = is_contained_v<T, gtl::signed_int_list>;

template <typename T>
inline constexpr bool is_ulonglongn_v =
    is_contained_v<T, gtl::vector_unsigned_longlong_list>;

template <typename T>
inline constexpr bool is_ugenlonglong_v =
    is_contained_v<T, gtl::unsigned_longlong_list>;

template <typename T>
inline constexpr bool is_longlongn_v =
    is_contained_v<T, gtl::vector_signed_longlong_list>;

template <typename T>
inline constexpr bool is_genlonglong_v =
    is_contained_v<T, gtl::signed_longlong_list>;

template <typename T>
inline constexpr bool is_igenlonginteger_v =
    is_contained_v<T, gtl::signed_long_integer_list>;

template <typename T>
inline constexpr bool is_ugenlonginteger_v =
    is_contained_v<T, gtl::unsigned_long_integer_list>;

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
inline constexpr bool is_geninteger8bit_v =
    is_gen_based_on_type_sizeof_v<T, 1, is_geninteger>;

template <typename T>
inline constexpr bool is_geninteger16bit_v =
    is_gen_based_on_type_sizeof_v<T, 2, is_geninteger>;

template <typename T>
inline constexpr bool is_geninteger32bit_v =
    is_gen_based_on_type_sizeof_v<T, 4, is_geninteger>;

template <typename T>
inline constexpr bool is_geninteger64bit_v =
    is_gen_based_on_type_sizeof_v<T, 8, is_geninteger>;

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
inline constexpr bool is_genptr_v =
    is_pointer_v<T> && is_gentype_v<remove_pointer_t<T>> &&
    is_address_space_compliant_v<T, gvl::nonconst_address_space_list>;

template <typename T>
inline constexpr bool is_nan_type_v = is_contained_v<T, gtl::nan_list>;

// nan_types
template <typename T, typename Enable = void> struct nan_types {
  // Nonsensical case for types implicitly convertible to scalar to avoid
  // templated overloads which are SFINAE'd out to cause compilation errors.
  using ret_type = void;
  using arg_type = int;
};

template <typename T>
struct nan_types<
    T, std::enable_if_t<is_contained_v<T, gtl::unsigned_short_list>, T>> {
  using ret_type = change_base_type_t<T, half>;
  using arg_type = find_same_size_type_t<gtl::scalar_unsigned_short_list, half>;
};

template <typename T>
struct nan_types<
    T, std::enable_if_t<is_contained_v<T, gtl::unsigned_int_list>, T>> {
  using ret_type = change_base_type_t<T, float>;
  using arg_type = find_same_size_type_t<gtl::scalar_unsigned_int_list, float>;
};

template <typename T>
struct nan_types<
    T,
    std::enable_if_t<is_contained_v<T, gtl::unsigned_long_integer_list>, T>> {
  using ret_type = change_base_type_t<T, double>;
  using arg_type =
      find_same_size_type_t<gtl::scalar_unsigned_long_integer_list, double>;
};

template <typename T> using nan_return_t = typename nan_types<T, T>::ret_type;

template <typename T>
using nan_argument_base_t = typename nan_types<T, T>::arg_type;

template <typename T>
using make_floating_point_t = make_type_t<T, gtl::scalar_floating_list>;

template <typename T>
using make_singed_integer_t = make_type_t<T, gtl::scalar_signed_integer_list>;

template <typename T>
using make_unsinged_integer_t =
    make_type_t<T, gtl::scalar_unsigned_integer_list>;

template <typename T, typename B, typename Enable = void>
struct convert_data_type_impl;

template <typename T, typename B>
struct convert_data_type_impl<T, B, std::enable_if_t<is_sgentype_v<T>, T>> {
  B operator()(T t) { return static_cast<B>(t); }
};

template <typename T, typename B>
struct convert_data_type_impl<T, B, std::enable_if_t<is_vgentype_v<T>, T>> {
  vec<B, T::size()> operator()(T t) { return t.template convert<B>(); }
};

template <typename T, typename B>
using convert_data_type = convert_data_type_impl<T, B, T>;

// TryToGetElementType<T>::type is T::element_type or T::value_type if those
// exist, otherwise T.
template <typename T> class TryToGetElementType {
  static T check(...);
  template <typename A> static typename A::element_type check(const A &);
  template <typename A> static typename A::value_type check(const A &);

public:
  using type = decltype(check(T()));
  static constexpr bool value = !std::is_same_v<T, type>;
};

template <typename To> struct PointerConverter {
  template <typename From> static To Convert(From *t) {
    return reinterpret_cast<To>(t);
  }

  template <typename From> static To Convert(From &t) {
    if constexpr (is_non_legacy_multi_ptr_v<From>) {
      return detail::cast_AS<To>(t.get_decorated());
    } else if constexpr (is_legacy_multi_ptr_v<From>) {
      return detail::cast_AS<To>(t.get());
    } else {
      // TODO find the better way to get the pointer to underlying data from vec
      // class
      return reinterpret_cast<To>(t.get());
    }
  }
};

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
struct PointerConverter<multi_ptr<ElementType, Space, DecorateAddress>> {
  template <typename From>
  static multi_ptr<ElementType, Space, DecorateAddress> Convert(From *t) {
    return address_space_cast<Space, DecorateAddress>(
        reinterpret_cast<remove_decoration_t<From *>>(t));
  }

  template <typename From>
  static multi_ptr<ElementType, Space, DecorateAddress> Convert(From &t) {
    return address_space_cast<Space, DecorateAddress>(
        reinterpret_cast<remove_decoration_t<decltype(t.get())>>(t.get()));
  }

  template <typename From>
  static multi_ptr<ElementType, Space, DecorateAddress>
  Convert(multi_ptr<ElementType, Space, DecorateAddress> &t) {
    return t;
  }
};

template <typename To, typename From> To ConvertNonVectorType(From &t) {
  if constexpr (is_pointer_v<From>)
    return PointerConverter<To>::Convert(t);
  else
    return static_cast<To>(t);
}

template <typename T, typename = void> struct mptr_or_vec_elem_type {
  using type = typename T::element_type;
};
template <typename ElementType, access::address_space Space,
          access::decorated IsDecorated>
struct mptr_or_vec_elem_type<
    multi_ptr<ElementType, Space, IsDecorated>,
    std::enable_if_t<IsDecorated == access::decorated::no ||
                     IsDecorated == access::decorated::yes>> {
  using type = typename multi_ptr<ElementType, Space, IsDecorated>::value_type;
};
template <typename ElementType, access::address_space Space,
          access::decorated IsDecorated>
struct mptr_or_vec_elem_type<const multi_ptr<ElementType, Space, IsDecorated>>
    : mptr_or_vec_elem_type<multi_ptr<ElementType, Space, IsDecorated>> {};

template <typename T>
using mptr_or_vec_elem_type_t = typename mptr_or_vec_elem_type<T>::type;

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

template <typename T>
using select_cl_scalar_float_t =
    select_apply_cl_scalar_t<T, std::false_type, sycl::opencl::cl_half,
                             sycl::opencl::cl_float, sycl::opencl::cl_double>;

// Use SFINAE so that std::complex specialization could be implemented in
// include/sycl/stl_wrappers/complex that would only be available if STL's
// <complex> is included by users.
template <typename T, typename = void> struct select_cl_scalar_complex_or_T {
  using type = T;
};

template <typename T>
using select_cl_scalar_complex_or_T_t =
    typename select_cl_scalar_complex_or_T<T>::type;

template <typename T>
using select_cl_scalar_integral_t =
    std::conditional_t<std::is_signed_v<T>,
                       select_cl_scalar_integral_signed_t<T>,
                       select_cl_scalar_integral_unsigned_t<T>>;

// select_cl_scalar_t picks corresponding cl_* type for input
// scalar T or returns T if T is not scalar.
template <typename T>
using select_cl_scalar_t = std::conditional_t<
    std::is_integral_v<T>, select_cl_scalar_integral_t<T>,
    std::conditional_t<
        std::is_floating_point_v<T>, select_cl_scalar_float_t<T>,
        // half and bfloat16 are special cases: they are implemented differently
        // on host and device and therefore might lower to different types
        std::conditional_t<
            is_half_v<T>, sycl::detail::half_impl::BIsRepresentationT,
            std::conditional_t<is_bfloat16_v<T>, T,
                               select_cl_scalar_complex_or_T_t<T>>>>>;

// select_cl_vector_or_scalar_or_ptr does cl_* type selection for element type
// of a vector type T, pointer type substitution, and scalar type substitution.
// If T is not vector, scalar, or pointer unmodified T is returned.
template <typename T, typename Enable = void>
struct select_cl_vector_or_scalar_or_ptr;

template <typename T>
struct select_cl_vector_or_scalar_or_ptr<
    T, typename std::enable_if_t<is_vgentype_v<T>>> {
  using type =
      // select_cl_scalar_t may return _Float16, so, we try to instantiate vec
      // class with _Float16 DataType, which is not expected there
      // So, leave vector<half, N> as-is
      vec<std::conditional_t<is_half_v<mptr_or_vec_elem_type_t<T>>,
                             mptr_or_vec_elem_type_t<T>,
                             select_cl_scalar_t<mptr_or_vec_elem_type_t<T>>>,
          T::size()>;
};

template <typename T>
struct select_cl_vector_or_scalar_or_ptr<
    T, typename std::enable_if_t<!is_vgentype_v<T> && !std::is_pointer_v<T>>> {
  using type = select_cl_scalar_t<T>;
};

template <typename T>
struct select_cl_vector_or_scalar_or_ptr<
    T, typename std::enable_if_t<!is_vgentype_v<T> && std::is_pointer_v<T>>> {
  using elem_ptr_type =
      typename select_cl_vector_or_scalar_or_ptr<std::remove_pointer_t<T>>::type
          *;
#ifdef __SYCL_DEVICE_ONLY__
  using type = typename DecoratedType<elem_ptr_type, deduce_AS<T>::value>::type;
#else
  using type = elem_ptr_type;
#endif
};

// select_cl_mptr_or_vector_or_scalar_or_ptr does cl_* type selection for type
// pointed by multi_ptr, for raw pointers, for element type of a vector type T,
// and does scalar type substitution.  If T is not mutlti_ptr or vector or
// scalar or pointer unmodified T is returned.
template <typename T, typename Enable = void>
struct select_cl_mptr_or_vector_or_scalar_or_ptr;

// this struct helps to use std::uint8_t instead of std::byte,
// which is not supported on device
template <typename T> struct TypeHelper {
  using RetType = T;
};

#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
template <> struct TypeHelper<std::byte> {
  using RetType = std::uint8_t;
};
#endif

template <typename T> struct TypeHelper<const T> {
  using RetType = const typename TypeHelper<T>::RetType;
};

template <typename T> struct TypeHelper<volatile T> {
  using RetType = volatile typename TypeHelper<T>::RetType;
};

template <typename T> struct TypeHelper<const volatile T> {
  using RetType = const volatile typename TypeHelper<T>::RetType;
};

template <typename T> using type_helper = typename TypeHelper<T>::RetType;

template <typename T>
struct select_cl_mptr_or_vector_or_scalar_or_ptr<
    T, typename std::enable_if_t<is_genptr_v<T> && !std::is_pointer_v<T>>> {
  using type = multi_ptr<typename select_cl_vector_or_scalar_or_ptr<
                             type_helper<mptr_or_vec_elem_type_t<T>>>::type,
                         T::address_space, access::decorated::yes>;
};

template <typename T>
struct select_cl_mptr_or_vector_or_scalar_or_ptr<
    T, typename std::enable_if_t<!is_genptr_v<T> || std::is_pointer_v<T>>> {
  using type = typename select_cl_vector_or_scalar_or_ptr<T>::type;
};

// All types converting shortcut.
template <typename T>
using SelectMatchingOpenCLType_t =
    typename select_cl_mptr_or_vector_or_scalar_or_ptr<T>::type;

// Converts T to OpenCL friendly
//
template <typename T> struct ConvertToOpenCLTypeImpl {
  using type = T;
};

#ifdef __SYCL_DEVICE_ONLY__
template <typename Type, int NumElements>
struct ConvertToOpenCLTypeImpl<vec<Type, NumElements>> {
  using type = typename vec<Type, NumElements>::vector_t;
};

template <int N> struct ConvertToOpenCLTypeImpl<Boolean<N>> {
  using type = typename Boolean<N>::vector_t;
};
template <> struct ConvertToOpenCLTypeImpl<Boolean<1>> {
  // Or should it be "int"?
  using type = Boolean<1>;
};
#endif

template <typename T> struct ConvertToOpenCLTypeImpl<T *> {
#ifdef __SYCL_DEVICE_ONLY__
  using type = typename DecoratedType<
      typename ConvertToOpenCLTypeImpl<remove_decoration_t<T>>::type,
      deduce_AS<T>::value>::type *;
#else
  using type = typename ConvertToOpenCLTypeImpl<T>::type *;
#endif
};

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
struct ConvertToOpenCLTypeImpl<multi_ptr<ElementType, Space, DecorateAddress>> {
  using type = typename DecoratedType<
      typename ConvertToOpenCLTypeImpl<ElementType>::type, Space>::type *;
};

template <typename T>
using ConvertToOpenCLType_t =
    typename ConvertToOpenCLTypeImpl<SelectMatchingOpenCLType_t<T>>::type;

// convertDataToType() function converts data from FROM type to TO type using
// 'as' method for vector type and copy otherwise.
template <typename FROM, typename TO>
typename std::enable_if_t<sizeof(TO) == sizeof(FROM), TO>
convertDataToType(FROM t) {
  if constexpr (is_vgentype_v<FROM> && is_vgentype_v<TO>)
    return t.template as<TO>();
  else
    return ConvertNonVectorType<TO>(t);
}

// Used for all, any and select relational built-in functions
template <typename T> inline constexpr T msbMask(T) {
  using UT = make_unsigned_t<T>;
  return T(UT(1) << (sizeof(T) * 8 - 1));
}

template <typename T> inline constexpr bool msbIsSet(const T x) {
  return (x & msbMask(x));
}

// SYCL 2020 4.17.9 (Relation functions), e.g. table 178
//
//  genbool isequal (genfloatf x, genfloatf y)
//  genbool isequal (genfloatd x, genfloatd y)
//
// TODO: marray support isn't implemented yet.
template <typename T>
using common_rel_ret_t =
    std::conditional_t<is_vgentype_v<T>, make_singed_integer_t<T>, bool>;

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

// Used for relational comparison built-in functions
template <typename T> struct RelationalReturnType {
#ifdef __SYCL_DEVICE_ONLY__
  static constexpr int N = GetNumElements<T>::value;
  using type = std::conditional_t<N == 1, bool, Boolean<N>>;
#else
  using type = common_rel_ret_t<T>;
#endif
};

// Type representing the internal return type of relational builtins.
template <typename T>
using internal_rel_ret_t = typename RelationalReturnType<T>::type;

// Used for any and all built-in functions
template <typename T> struct RelationalTestForSignBitType {
#ifdef __SYCL_DEVICE_ONLY__
  using return_type = bool;
  static constexpr int N = GetNumElements<T>::value;
  using argument_type = std::conditional_t<N == 1, bool, detail::Boolean<N>>;
#else
  using return_type = int;
  using argument_type = T;
#endif
};

template <typename T>
using rel_sign_bit_test_ret_t =
    typename RelationalTestForSignBitType<T>::return_type;

template <typename T>
using rel_sign_bit_test_arg_t =
    typename RelationalTestForSignBitType<T>::argument_type;

template <typename T, typename Enable = void> struct RelConverter {
  using R = internal_rel_ret_t<T>;
#ifdef __SYCL_DEVICE_ONLY__
  using value_t = bool;
#else
  using value_t = R;
#endif

  static R apply(value_t value) { return value; }
};

template <typename T>
struct RelConverter<T,
                    typename std::enable_if_t<TryToGetElementType<T>::value>> {
  static const int N = T::size();
#ifdef __SYCL_DEVICE_ONLY__
  using bool_t = typename Boolean<N>::vector_t;
  using ret_t = common_rel_ret_t<T>;
#else
  using bool_t = Boolean<N>;
  using ret_t = internal_rel_ret_t<T>;
#endif

  static ret_t apply(bool_t value) {
#ifdef __SYCL_DEVICE_ONLY__
    typename ret_t::vector_t result(0);
    for (size_t I = 0; I < N; ++I) {
      result[I] = value[I];
    }
    return result;
#else
    return value;
#endif
  }
};

template <typename T> static constexpr T max_v() {
  return (std::numeric_limits<T>::max)();
}

template <typename T> static constexpr T min_v() {
  return (std::numeric_limits<T>::min)();
}

template <typename T> static constexpr T quiet_NaN() {
  return std::numeric_limits<T>::quiet_NaN();
}

// is_same_vector_size
template <int FirstSize, typename... Args> class is_same_vector_size_impl;

template <int FirstSize, typename T, typename... Args>
class is_same_vector_size_impl<FirstSize, T, Args...> {
  using CurrentT = detail::remove_pointer_t<T>;
  static constexpr int Size = vector_size<CurrentT>::value;
  static constexpr bool IsSizeEqual = (Size == FirstSize);

public:
  static constexpr bool value =
      IsSizeEqual ? is_same_vector_size_impl<FirstSize, Args...>::value : false;
};

template <int FirstSize>
class is_same_vector_size_impl<FirstSize> : public std::true_type {};

template <typename T, typename... Args> class is_same_vector_size {
  using CurrentT = remove_pointer_t<T>;
  static constexpr int Size = vector_size<CurrentT>::value;

public:
  static constexpr bool value = is_same_vector_size_impl<Size, Args...>::value;
};

// check_vector_size
template <typename... Args> inline void check_vector_size() {
  static_assert(is_same_vector_size<Args...>::value,
                "The built-in function arguments must [point to|have] types "
                "with the same number of elements.");
}

} // namespace detail
} // namespace _V1
} // namespace sycl
