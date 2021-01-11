//==----------- type_list.hpp - SYCL list of types utils -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/stl_type_traits.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

template <typename T> using head_t = typename T::head;

template <typename T> using tail_t = typename T::tail;

// type_list
template <typename... T> struct type_list;

using empty_type_list = type_list<>;

template <typename T>
struct is_empty_type_list
    : conditional_t<std::is_same<T, empty_type_list>::value, std::true_type,
                    std::false_type> {};

template <> struct type_list<> {};

template <typename H, typename... T> struct type_list<H, T...> {
  using head = H;
  using tail = type_list<T...>;
};

template <typename H, typename... T, typename... T2>
struct type_list<type_list<H, T...>, T2...> {
private:
  using remainder = tail_t<type_list<H>>;
  static constexpr bool has_remainder = !is_empty_type_list<remainder>::value;
  using without_remainder = type_list<T..., T2...>;
  using with_remainder = type_list<remainder, T..., T2...>;

public:
  using head = head_t<type_list<H>>;
  using tail = conditional_t<has_remainder, with_remainder, without_remainder>;
};

// is_contained
template <typename T, typename L>
struct is_contained
    : conditional_t<std::is_same<remove_cv_t<T>, head_t<L>>::value,
                    std::true_type, is_contained<T, tail_t<L>>> {};

template <typename T>
struct is_contained<T, empty_type_list> : std::false_type {};

// value_list
template <typename T, T... V> struct value_list;

template <typename T, T H, T... V> struct value_list<T, H, V...> {
  static constexpr T head = H;
  using tail = value_list<T, V...>;
};

template <typename T> struct value_list<T> {};

// is_contained_value
template <typename T, T V, typename TL>
struct is_contained_value
    : conditional_t<V == TL::head, std::true_type,
                    is_contained_value<T, V, tail_t<TL>>> {};

template <typename T, T V>
struct is_contained_value<T, V, value_list<T>> : std::false_type {};

//  address_space_list
template <access::address_space... V>
using address_space_list = value_list<access::address_space, V...>;

template <access::address_space AS, typename VL>
using is_one_of_spaces = is_contained_value<access::address_space, AS, VL>;

// size type predicates
template <typename T1, typename T2>
struct is_type_size_equal : bool_constant<(sizeof(T1) == sizeof(T2))> {};

template <typename T1, typename T2>
struct is_type_size_greater : bool_constant<(sizeof(T1) > sizeof(T2))> {};

template <typename T1, typename T2>
struct is_type_size_double_of
    : bool_constant<(sizeof(T1) == (sizeof(T2) * 2))> {};

template <typename T1, typename T2>
struct is_type_size_less : bool_constant<(sizeof(T1) < sizeof(T2))> {};

template <typename T1, typename T2>
struct is_type_size_half_of : bool_constant<(sizeof(T1) == (sizeof(T2) / 2))> {
};

// find required type
template <typename TL, template <typename, typename> class C, typename T>
struct find_type {
  using head = head_t<TL>;
  using tail = typename find_type<tail_t<TL>, C, T>::type;
  using type = conditional_t<C<head, T>::value, head, tail>;
};

template <template <typename, typename> class C, typename T>
struct find_type<empty_type_list, C, T> {
  using type = void;
};

template <typename TL, template <typename, typename> class C, typename T>
using find_type_t = typename find_type<TL, C, T>::type;

template <typename TL, typename T>
using find_same_size_type_t = find_type_t<TL, is_type_size_equal, T>;

template <typename TL, typename T>
using find_smaller_type_t = find_type_t<TL, is_type_size_less, T>;

template <typename TL, typename T>
using find_larger_type_t = find_type_t<TL, is_type_size_greater, T>;

template <typename TL, typename T>
using find_twice_as_small_type_t = find_type_t<TL, is_type_size_half_of, T>;

template <typename TL, typename T>
using find_twice_as_large_type_t = find_type_t<TL, is_type_size_double_of, T>;

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
