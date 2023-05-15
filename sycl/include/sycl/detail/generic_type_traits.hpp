//==----------- generic_type_traits - SYCL type traits ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_types.hpp>
#include <sycl/access/access.hpp>
#include <sycl/aliases.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/generic_type_lists.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/half_type.hpp>
#include <sycl/multi_ptr.hpp>

#include <complex>
#include <limits>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

template <typename T> using is_floatn = is_contained<T, gtl::vector_float_list>;

template <typename T> using is_genfloatf = is_contained<T, gtl::float_list>;

template <typename T>
using is_svgenfloatf = is_contained<T, gtl::scalar_vector_float_list>;

template <typename T>
using is_doublen = is_contained<T, gtl::vector_double_list>;

template <typename T> using is_genfloatd = is_contained<T, gtl::double_list>;

template <typename T>
using is_svgenfloatd = is_contained<T, gtl::scalar_vector_double_list>;

template <typename T> using is_halfn = is_contained<T, gtl::vector_half_list>;

template <typename T> using is_genfloath = is_contained<T, gtl::half_list>;

template <typename T> using is_half = is_contained<T, gtl::scalar_half_list>;

template <typename T>
using is_svgenfloath = is_contained<T, gtl::scalar_vector_half_list>;

template <typename T> using is_genfloat = is_contained<T, gtl::floating_list>;

template <typename T>
using is_sgenfloat = is_contained<T, gtl::scalar_floating_list>;

template <typename T>
using is_vgenfloat = is_contained<T, gtl::vector_floating_list>;

template <typename T>
using is_svgenfloat = is_contained<T, gtl::scalar_vector_floating_list>;

template <typename T>
using is_mgenfloat = std::bool_constant<
    std::is_same_v<T, sycl::marray<marray_element_t<T>, T::size()>> &&
    is_svgenfloat<marray_element_t<T>>::value>;

template <typename T>
using is_gengeofloat = is_contained<T, gtl::geo_float_list>;

template <typename T>
using is_gengeodouble = is_contained<T, gtl::geo_double_list>;

template <typename T>
using is_gengeomarrayfloat = is_contained<T, gtl::marray_geo_float_list>;

template <typename T>
using is_gengeomarray = is_contained<T, gtl::marray_geo_list>;

template <typename T> using is_gengeohalf = is_contained<T, gtl::geo_half_list>;

template <typename T>
using is_vgengeofloat = is_contained<T, gtl::vector_geo_float_list>;

template <typename T>
using is_vgengeodouble = is_contained<T, gtl::vector_geo_double_list>;

template <typename T>
using is_vgengeohalf = is_contained<T, gtl::vector_geo_half_list>;

template <typename T> using is_sgengeo = is_contained<T, gtl::scalar_geo_list>;

template <typename T> using is_vgengeo = is_contained<T, gtl::vector_geo_list>;

template <typename T>
using is_gencrossfloat = is_contained<T, gtl::cross_float_list>;

template <typename T>
using is_gencrossdouble = is_contained<T, gtl::cross_double_list>;

template <typename T>
using is_gencrosshalf = is_contained<T, gtl::cross_half_list>;

template <typename T>
using is_gencross = is_contained<T, gtl::cross_floating_list>;

template <typename T>
using is_gencrossmarray = is_contained<T, gtl::cross_marray_list>;

template <typename T>
using is_charn = is_contained<T, gtl::vector_default_char_list>;

template <typename T>
using is_scharn = is_contained<T, gtl::vector_signed_char_list>;

template <typename T>
using is_ucharn = is_contained<T, gtl::vector_unsigned_char_list>;

template <typename T>
using is_igenchar = is_contained<T, gtl::signed_char_list>;

template <typename T>
using is_ugenchar = is_contained<T, gtl::unsigned_char_list>;

template <typename T> using is_genchar = is_contained<T, gtl::char_list>;

template <typename T>
using is_shortn = is_contained<T, gtl::vector_signed_short_list>;

template <typename T>
using is_genshort = is_contained<T, gtl::signed_short_list>;

template <typename T>
using is_ushortn = is_contained<T, gtl::vector_unsigned_short_list>;

template <typename T>
using is_ugenshort = is_contained<T, gtl::unsigned_short_list>;

template <typename T>
using is_uintn = is_contained<T, gtl::vector_unsigned_int_list>;

template <typename T>
using is_ugenint = is_contained<T, gtl::unsigned_int_list>;

template <typename T>
using is_intn = is_contained<T, gtl::vector_signed_int_list>;

template <typename T> using is_genint = is_contained<T, gtl::signed_int_list>;

template <typename T>
using is_ulonglongn = is_contained<T, gtl::vector_unsigned_longlong_list>;

template <typename T>
using is_ugenlonglong = is_contained<T, gtl::unsigned_longlong_list>;

template <typename T>
using is_longlongn = is_contained<T, gtl::vector_signed_longlong_list>;

template <typename T>
using is_genlonglong = is_contained<T, gtl::signed_longlong_list>;

template <typename T>
using is_igenlonginteger = is_contained<T, gtl::signed_long_integer_list>;

template <typename T>
using is_ugenlonginteger = is_contained<T, gtl::unsigned_long_integer_list>;

template <typename T> using is_geninteger = is_contained<T, gtl::integer_list>;

template <typename T>
using is_igeninteger = is_contained<T, gtl::signed_integer_list>;

template <typename T>
using is_ugeninteger = is_contained<T, gtl::unsigned_integer_list>;

template <typename T>
using is_sgeninteger = is_contained<T, gtl::scalar_integer_list>;

template <typename T>
using is_vgeninteger = is_contained<T, gtl::vector_integer_list>;

template <typename T>
using is_sigeninteger = is_contained<T, gtl::scalar_signed_integer_list>;

template <typename T>
using is_sugeninteger = is_contained<T, gtl::scalar_unsigned_integer_list>;

template <typename T>
using is_vigeninteger = is_contained<T, gtl::vector_signed_integer_list>;

template <typename T>
using is_vugeninteger = is_contained<T, gtl::vector_unsigned_integer_list>;

template <typename T> using is_genbool = is_contained<T, gtl::bool_list>;

template <typename T> using is_gentype = is_contained<T, gtl::basic_list>;

template <typename T>
using is_vgentype = is_contained<T, gtl::vector_basic_list>;

template <typename T>
using is_sgentype = is_contained<T, gtl::scalar_basic_list>;

template <typename T>
using is_igeninteger8bit = is_gen_based_on_type_sizeof<T, 1, is_igeninteger>;

template <typename T>
using is_igeninteger16bit = is_gen_based_on_type_sizeof<T, 2, is_igeninteger>;

template <typename T>
using is_igeninteger32bit = is_gen_based_on_type_sizeof<T, 4, is_igeninteger>;

template <typename T>
using is_igeninteger64bit = is_gen_based_on_type_sizeof<T, 8, is_igeninteger>;

template <typename T>
using is_ugeninteger8bit = is_gen_based_on_type_sizeof<T, 1, is_ugeninteger>;

template <typename T>
using is_ugeninteger16bit = is_gen_based_on_type_sizeof<T, 2, is_ugeninteger>;

template <typename T>
using is_ugeninteger32bit = is_gen_based_on_type_sizeof<T, 4, is_ugeninteger>;

template <typename T>
using is_ugeninteger64bit = is_gen_based_on_type_sizeof<T, 8, is_ugeninteger>;

template <typename T>
using is_geninteger8bit = is_gen_based_on_type_sizeof<T, 1, is_geninteger>;

template <typename T>
using is_geninteger16bit = is_gen_based_on_type_sizeof<T, 2, is_geninteger>;

template <typename T>
using is_geninteger32bit = is_gen_based_on_type_sizeof<T, 4, is_geninteger>;

template <typename T>
using is_geninteger64bit = is_gen_based_on_type_sizeof<T, 8, is_geninteger>;

template <typename T>
using is_genintptr = std::bool_constant<
    is_pointer<T>::value && is_genint<remove_pointer_t<T>>::value &&
    is_address_space_compliant<T, gvl::nonconst_address_space_list>::value>;

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
using is_genintptr_marray = std::bool_constant<
    std::is_same_v<T, sycl::marray<marray_element_t<T>, T::size()>> &&
    is_genint<marray_element_t<remove_pointer_t<T>>>::value &&
    is_address_space_compliant<multi_ptr<T, AddressSpace, IsDecorated>,
                               gvl::nonconst_address_space_list>::value &&
    (IsDecorated == access::decorated::yes ||
     IsDecorated == access::decorated::no)>;

template <typename T>
using is_genfloatptr = std::bool_constant<
    is_pointer<T>::value && is_genfloat<remove_pointer_t<T>>::value &&
    is_address_space_compliant<T, gvl::nonconst_address_space_list>::value>;

template <typename T, access::address_space AddressSpace,
          access::decorated IsDecorated>
using is_genfloatptr_marray = std::bool_constant<
    is_mgenfloat<T>::value &&
    is_address_space_compliant<multi_ptr<T, AddressSpace, IsDecorated>,
                               gvl::nonconst_address_space_list>::value &&
    (IsDecorated == access::decorated::yes ||
     IsDecorated == access::decorated::no)>;

template <typename T>
using is_genptr = std::bool_constant<
    is_pointer<T>::value && is_gentype<remove_pointer_t<T>>::value &&
    is_address_space_compliant<T, gvl::nonconst_address_space_list>::value>;

template <typename T> using is_nan_type = is_contained<T, gtl::nan_list>;

// nan_types
template <typename T, typename Enable = void> struct nan_types;

template <typename T>
struct nan_types<
    T, std::enable_if_t<is_contained<T, gtl::unsigned_short_list>::value, T>> {
  using ret_type = change_base_type_t<T, half>;
  using arg_type = find_same_size_type_t<gtl::scalar_unsigned_short_list, half>;
};

template <typename T>
struct nan_types<
    T, std::enable_if_t<is_contained<T, gtl::unsigned_int_list>::value, T>> {
  using ret_type = change_base_type_t<T, float>;
  using arg_type = find_same_size_type_t<gtl::scalar_unsigned_int_list, float>;
};

template <typename T>
struct nan_types<
    T, std::enable_if_t<is_contained<T, gtl::unsigned_long_integer_list>::value,
                        T>> {
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
struct convert_data_type_impl<T, B,
                              std::enable_if_t<is_sgentype<T>::value, T>> {
  B operator()(T t) { return static_cast<B>(t); }
};

template <typename T, typename B>
struct convert_data_type_impl<T, B,
                              std::enable_if_t<is_vgentype<T>::value, T>> {
  vec<B, T::size()> operator()(T t) { return t.template convert<B>(); }
};

template <typename T, typename B>
using convert_data_type = convert_data_type_impl<T, B, T>;

// TryToGetPointerT<T>::type is T::pointer_t (legacy) or T::pointer if those
// exist, otherwise T.
template <typename T> class TryToGetPointerT {
  static T check(...);
  template <typename A> static typename A::pointer_t check(const A &);
  template <typename A> static typename A::pointer check(const A &);

public:
  using type = decltype(check(T()));
  static constexpr bool value =
      std::is_pointer_v<T> || !std::is_same_v<T, type>;
};

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

// TryToGetVectorT<T>::type is T::vector_t if that exists, otherwise T.
template <typename T> class TryToGetVectorT {
  static T check(...);
  template <typename A> static typename A::vector_t check(const A &);

public:
  using type = decltype(check(T()));
  static constexpr bool value = !std::is_same_v<T, type>;
};

// Try to get pointer_t (if pointer_t indicates on the type with_remainder
// vector_t creates a pointer type on vector_t), otherwise T
template <typename T> class TryToGetPointerVecT {
  static T check(...);
  template <typename A>
  static typename DecoratedType<
      typename TryToGetVectorT<typename TryToGetElementType<A>::type>::type,
      A::address_space>::type *
  check(const A &);
  template <typename A>
  static typename TryToGetVectorT<A>::type *check(const A *);

public:
  using type = decltype(check(T()));
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

template <typename To, typename From,
          typename = typename std::enable_if_t<TryToGetPointerT<From>::value>>
To ConvertNonVectorType(From &t) {
  return PointerConverter<To>::Convert(t);
}

template <typename To, typename From> To ConvertNonVectorType(From *t) {
  return PointerConverter<To>::Convert(t);
}

template <typename To, typename From>
typename std::enable_if_t<!TryToGetPointerT<From>::value, To>
ConvertNonVectorType(From &t) {
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

template <typename T>
using select_cl_scalar_complex_or_T_t = std::conditional_t<
    std::is_same_v<T, std::complex<float>>, __spv::complex_float,
    std::conditional_t<std::is_same_v<T, std::complex<double>>,
                       __spv::complex_double,
                       std::conditional_t<std::is_same_v<T, std::complex<half>>,
                                          __spv::complex_half, T>>>;

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
        // half is a special case: it is implemented differently on
        // host and device and therefore, might lower to different
        // types
        std::conditional_t<is_half<T>::value,
                           sycl::detail::half_impl::BIsRepresentationT,
                           select_cl_scalar_complex_or_T_t<T>>>>;

// select_cl_vector_or_scalar_or_ptr does cl_* type selection for element type
// of a vector type T, pointer type substitution, and scalar type substitution.
// If T is not vector, scalar, or pointer unmodified T is returned.
template <typename T, typename Enable = void>
struct select_cl_vector_or_scalar_or_ptr;

template <typename T>
struct select_cl_vector_or_scalar_or_ptr<
    T, typename std::enable_if_t<is_vgentype<T>::value>> {
  using type =
      // select_cl_scalar_t returns _Float16, so, we try to instantiate vec
      // class with _Float16 DataType, which is not expected there
      // So, leave vector<half, N> as-is
      vec<std::conditional_t<is_half<mptr_or_vec_elem_type_t<T>>::value,
                             mptr_or_vec_elem_type_t<T>,
                             select_cl_scalar_t<mptr_or_vec_elem_type_t<T>>>,
          T::size()>;
};

template <typename T>
struct select_cl_vector_or_scalar_or_ptr<
    T, typename std::enable_if_t<!is_vgentype<T>::value &&
                                 !std::is_pointer_v<T>>> {
  using type = select_cl_scalar_t<T>;
};

template <typename T>
struct select_cl_vector_or_scalar_or_ptr<
    T,
    typename std::enable_if_t<!is_vgentype<T>::value && std::is_pointer_v<T>>> {
  using elem_ptr_type = typename select_cl_vector_or_scalar_or_ptr<
      std::remove_pointer_t<T>>::type *;
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
    T,
    typename std::enable_if_t<is_genptr<T>::value && !std::is_pointer_v<T>>> {
  using type = multi_ptr<typename select_cl_vector_or_scalar_or_ptr<
                             type_helper<mptr_or_vec_elem_type_t<T>>>::type,
                         T::address_space, access::decorated::yes>;
};

template <typename T>
struct select_cl_mptr_or_vector_or_scalar_or_ptr<
    T,
    typename std::enable_if_t<!is_genptr<T>::value || std::is_pointer_v<T>>> {
  using type = typename select_cl_vector_or_scalar_or_ptr<T>::type;
};

// All types converting shortcut.
template <typename T>
using SelectMatchingOpenCLType_t =
    typename select_cl_mptr_or_vector_or_scalar_or_ptr<T>::type;

// Converts T to OpenCL friendly
//
template <typename T /* MatchingOpencCLTypeT */>
using ConvertToOpenCLTypeImpl_t = std::conditional_t<
    TryToGetVectorT<T>::value, typename TryToGetVectorT<T>::type,
    std::conditional_t<TryToGetPointerT<T>::value,
                       typename TryToGetPointerVecT<T>::type, T>>;
template <typename T>
using ConvertToOpenCLType_t =
    ConvertToOpenCLTypeImpl_t<SelectMatchingOpenCLType_t<T>>;

// convertDataToType() function converts data from FROM type to TO type using
// 'as' method for vector type and copy otherwise.
template <typename FROM, typename TO>
typename std::enable_if_t<is_vgentype<FROM>::value && is_vgentype<TO>::value &&
                              sizeof(TO) == sizeof(FROM),
                          TO>
convertDataToType(FROM t) {
  return t.template as<TO>();
}

template <typename FROM, typename TO>
typename std::enable_if_t<!(is_vgentype<FROM>::value &&
                            is_vgentype<TO>::value) &&
                              sizeof(TO) == sizeof(FROM),
                          TO>
convertDataToType(FROM t) {
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

#if defined(SYCL2020_CONFORMANT_APIS) && SYCL_LANGUAGE_VERSION >= 202001
// SYCL 2020 4.17.9 (Relation functions), e.g. table 178
//
//  genbool isequal (genfloatf x, genfloatf y)
//  genbool isequal (genfloatd x, genfloatd y)
//
// TODO: marray support isn't implemented yet.
template <typename T>
using common_rel_ret_t =
    std::conditional_t<is_vgentype<T>::value, make_singed_integer_t<T>, bool>;

// TODO: Remove this when common_rel_ret_t is promoted.
template <typename T>
using internal_host_rel_ret_t =
    std::conditional_t<is_vgentype<T>::value, make_singed_integer_t<T>, int>;
#else
// SYCL 1.2.1 4.13.7 (Relation functions), e.g.
//
//   igeninteger32bit isequal (genfloatf x, genfloatf y)
//   igeninteger64bit isequal (genfloatd x, genfloatd y)
//
// However, we have pre-existing bug so
//
//   igeninteger32bit isequal (genfloatd x, genfloatd y)
//
// Fixing it would be an ABI-breaking change so isn't done.
template <typename T>
using common_rel_ret_t =
    std::conditional_t<is_vgentype<T>::value, make_singed_integer_t<T>, int>;
template <typename T> using internal_host_rel_ret_t = common_rel_ret_t<T>;
#endif

// forward declaration
template <int N> struct Boolean;

// Try to get vector element count or 1 otherwise
template <typename T, typename Enable = void> struct TryToGetNumElements;

template <typename T>
struct TryToGetNumElements<
    T, typename std::enable_if_t<TryToGetVectorT<T>::value>> {
  static constexpr int value = T::size();
};
template <typename T>
struct TryToGetNumElements<
    T, typename std::enable_if_t<!TryToGetVectorT<T>::value>> {
  static constexpr int value = 1;
};

// Used for relational comparison built-in functions
template <typename T> struct RelationalReturnType {
#ifdef __SYCL_DEVICE_ONLY__
  using type = Boolean<TryToGetNumElements<T>::value>;
#else
  // After changing the return type of scalar relational operations to boolean
  // we keep the old representation of the internal implementation of the
  // host-side builtins to avoid ABI-breaks.
  // TODO: Use common_rel_ret_t when ABI break is allowed and the boolean return
  //       type for relationals are promoted out of SYCL2020_CONFORMANT_APIS.
  //       The scalar relational builtins in
  //       sycl/source/detail/builtins_relational.cpp should likewise be updated
  //       to return boolean values.
  using type = internal_host_rel_ret_t<T>;
#endif
};

// Type representing the internal return type of relational builtins.
template <typename T>
using internal_rel_ret_t = typename RelationalReturnType<T>::type;

// Used for any and all built-in functions
template <typename T> struct RelationalTestForSignBitType {
#ifdef __SYCL_DEVICE_ONLY__
  using return_type = detail::Boolean<1>;
  using argument_type = detail::Boolean<TryToGetNumElements<T>::value>;
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

template <typename T, typename Enable = void> struct RelConverter;

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

template <typename T>
struct RelConverter<T,
                    typename std::enable_if_t<!TryToGetElementType<T>::value>> {
  using R = internal_rel_ret_t<T>;
#ifdef __SYCL_DEVICE_ONLY__
  using value_t = bool;
#else
  using value_t = R;
#endif

  static R apply(value_t value) { return value; }
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
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
