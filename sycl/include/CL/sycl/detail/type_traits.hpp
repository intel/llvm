//==----------------- type_traits.hpp - SYCL type traits -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/generic_type_lists.hpp>
#include <CL/sycl/detail/stl_type_traits.hpp>
#include <CL/sycl/detail/type_list.hpp>
#include <CL/sycl/half_type.hpp>

#include <type_traits>

namespace cl {
namespace sycl {

// Forward declaration
template <typename ElementType, access::address_space Space> class multi_ptr;

namespace detail {

// Contains a type that is the base type for a scalar or vector type
template <typename T> struct get_base_type { using type = T; };

template <typename T, int N> struct get_base_type<vec<T, N>> {
  using type = T;
};

template <typename T> using get_base_type_t = typename get_base_type<T>::type;

// change_base_type_t
template <typename T, typename B> struct change_base_type { using type = B; };

template <typename T, int N, typename B> struct change_base_type<vec<T, N>, B> {
  using type = vec<B, N>;
};

template <typename T, typename B>
using change_base_type_t = typename change_base_type<T, B>::type;

// Applies the same the cv-qualifiers from T type to R type
template <typename T, typename R> struct copy_cv_qualifiers_impl {
  using type = R;
};

template <typename T, typename R> struct copy_cv_qualifiers_impl<const T, R> {
  using type = const R;
};

template <typename T, typename R>
struct copy_cv_qualifiers_impl<volatile T, R> {
  using type = volatile R;
};

template <typename T, typename R>
struct copy_cv_qualifiers_impl<const volatile T, R> {
  using type = const volatile R;
};

template <typename T, typename R> struct copy_cv_qualifiers {
  using type = typename copy_cv_qualifiers_impl<T, remove_cv_t<R>>::type;
};

template <typename T, typename R>
using copy_cv_qualifiers_t = typename copy_cv_qualifiers<T, R>::type;

// make_signed with support SYCL vec class

template <typename T, typename Enable = void> struct make_signed_impl;

template <typename T>
using make_signed_impl_t = typename make_signed_impl<T, T>::type;

template <typename T>
struct make_signed_impl<
    T, enable_if_t<is_contained<T, gtl::scalar_integer_list>::value, T>> {
  using type = typename std::make_signed<T>::type;
};

template <typename T>
struct make_signed_impl<
    T, enable_if_t<is_contained<T, gtl::vector_integer_list>::value, T>> {
  using base_type = make_signed_impl_t<get_base_type_t<T>>;
  using type = change_base_type_t<T, base_type>;
};

// TODO Delete this specialization after solving the problems in the test
// infrastructure.
template <typename T>
struct make_signed_impl<
    T, enable_if_t<!is_contained<T, gtl::integer_list>::value, T>> {
  using type = T;
};

template <typename T> struct make_signed {
  using new_type_wo_cv_qualifiers = make_signed_impl_t<remove_cv_t<T>>;
  using type = copy_cv_qualifiers_t<T, new_type_wo_cv_qualifiers>;
};

template <typename T> using make_signed_t = typename make_signed<T>::type;

// make_unsigned with support SYCL vec class
template <typename T, typename Enable = void> struct make_unsigned_impl;

template <typename T>
using make_unsigned_impl_t = typename make_unsigned_impl<T, T>::type;

template <typename T>
struct make_unsigned_impl<
    T, enable_if_t<is_contained<T, gtl::scalar_integer_list>::value, T>> {
  using type = typename std::make_unsigned<T>::type;
};

template <typename T>
struct make_unsigned_impl<
    T, enable_if_t<is_contained<T, gtl::vector_integer_list>::value, T>> {
  using base_type = make_unsigned_impl_t<get_base_type_t<T>>;
  using type = change_base_type_t<T, base_type>;
};

// TODO Delete this specialization after solving the problems in the test
// infrastructure.
template <typename T>
struct make_unsigned_impl<
    T, enable_if_t<!is_contained<T, gtl::integer_list>::value, T>> {
  using type = T;
};

template <typename T> struct make_unsigned {
  using new_type_wo_cv_qualifiers = make_unsigned_impl_t<remove_cv_t<T>>;
  using type = copy_cv_qualifiers_t<T, new_type_wo_cv_qualifiers>;
};

template <typename T> using make_unsigned_t = typename make_unsigned<T>::type;

// Checks that sizeof base type of T equal N and T satisfies S<T>::value
template <typename T, int N, template <typename> class S>
using is_gen_based_on_type_sizeof =
    bool_constant<S<T>::value && (sizeof(get_base_type_t<T>) == N)>;

// is_integral
template <typename T>
struct is_integral : std::is_integral<get_base_type_t<T>> {};

// is_floating_point
template <typename T>
struct is_floating_point_impl : std::is_floating_point<T> {};

template <> struct is_floating_point_impl<half> : std::true_type {};

template <typename T>
struct is_floating_point
    : is_floating_point_impl<remove_cv_t<get_base_type_t<T>>> {};

// is_arithmetic
template <typename T>
struct is_arithmetic
    : bool_constant<is_integral<T>::value || is_floating_point<T>::value> {};

// is_pointer
template <typename T> struct is_pointer_impl : std::false_type {};

template <typename T> struct is_pointer_impl<T *> : std::true_type {};

template <typename T, access::address_space Space>
struct is_pointer_impl<multi_ptr<T, Space>> : std::true_type {};

template <typename T> struct is_pointer : is_pointer_impl<remove_cv_t<T>> {};

// remove_pointer_t
template <typename T> struct remove_pointer_impl { using type = T; };

template <typename T> struct remove_pointer_impl<T *> { using type = T; };

template <typename T, access::address_space Space>
struct remove_pointer_impl<multi_ptr<T, Space>> {
  using type = T;
};

template <typename T>
struct remove_pointer : remove_pointer_impl<remove_cv_t<T>> {};

template <typename T> using remove_pointer_t = typename remove_pointer<T>::type;

// is_address_space_compliant
template <typename T, typename SpaceList>
struct is_address_space_compliant_impl : std::false_type {};

template <typename T, typename SpaceList>
struct is_address_space_compliant_impl<T *, SpaceList> : std::true_type {};

template <typename T, typename SpaceList, access::address_space Space>
struct is_address_space_compliant_impl<multi_ptr<T, Space>, SpaceList>
    : bool_constant<is_one_of_spaces<Space, SpaceList>::value> {};

template <typename T, typename SpaceList>
struct is_address_space_compliant
    : is_address_space_compliant_impl<remove_cv_t<T>, SpaceList> {};

// make_type_t
template <typename T, typename TL> struct make_type_impl {
  using type = find_same_size_type_t<TL, T>;
};

template <typename T, int N, typename TL> struct make_type_impl<vec<T, N>, TL> {
  using scalar_type = typename make_type_impl<T, TL>::type;
  using type = vec<scalar_type, N>;
};

template <typename T, typename TL>
using make_type_t = typename make_type_impl<T, TL>::type;

// nan_types
template <typename T, typename Enable = void> struct nan_types;

template <typename T>
struct nan_types<
    T, enable_if_t<is_contained<T, gtl::unsigned_short_list>::value, T>> {
  using ret_type = change_base_type_t<T, half>;
  using arg_type = find_same_size_type_t<gtl::scalar_unsigned_short_list, half>;
};

template <typename T>
struct nan_types<
    T, enable_if_t<is_contained<T, gtl::unsigned_int_list>::value, T>> {
  using ret_type = change_base_type_t<T, float>;
  using arg_type = find_same_size_type_t<gtl::scalar_unsigned_int_list, float>;
};

template <typename T>
struct nan_types<
    T,
    enable_if_t<is_contained<T, gtl::unsigned_long_integer_list>::value, T>> {
  using ret_type = change_base_type_t<T, double>;
  using arg_type =
      find_same_size_type_t<gtl::scalar_unsigned_long_integer_list, double>;
};

// make_larger_t
template <typename T, typename Enable = void> struct make_larger_impl;
template <typename T>
struct make_larger_impl<
    T, enable_if_t<is_contained<T, gtl::scalar_floating_list>::value, T>> {
  using type = find_twice_as_large_type_t<gtl::scalar_floating_list, T>;
};

template <typename T>
struct make_larger_impl<
    T,
    enable_if_t<is_contained<T, gtl::scalar_signed_integer_list>::value, T>> {
  using type = find_twice_as_large_type_t<gtl::scalar_signed_integer_list, T>;
};

template <typename T>
struct make_larger_impl<
    T,
    enable_if_t<is_contained<T, gtl::scalar_unsigned_integer_list>::value, T>> {
  using type = find_twice_as_large_type_t<gtl::scalar_unsigned_integer_list, T>;
};

template <typename T, int N> struct make_larger_impl<vec<T, N>, vec<T, N>> {
  using base_type = get_base_type_t<vec<T, N>>;
  using upper_type = typename make_larger_impl<base_type, base_type>::type;
  using new_type = vec<upper_type, N>;
  static constexpr bool found = !std::is_same<upper_type, void>::value;
  using type = conditional_t<found, new_type, void>;
};

template <typename T> struct make_larger {
  using type = typename make_larger_impl<T, T>::type;
};

template <typename T> using make_larger_t = typename make_larger<T>::type;

} // namespace detail
} // namespace sycl
} // namespace cl
