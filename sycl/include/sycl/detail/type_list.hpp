//==----------- type_list.hpp - SYCL list of types utils -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp> // for address_space

#include <type_traits> // for bool_constant, conditional_t, fals...

#include <sycl/detail/boost/mp11/algorithm.hpp>
#include <sycl/detail/boost/mp11/set.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

template <class... T> using type_list = boost::mp11::mp_list<T...>;

using empty_type_list = type_list<>;

template <typename T>
using is_empty_type_list = std::is_same<T, empty_type_list>;

// is_contained
template <typename T, typename TypeList>
using is_contained =
    boost::mp11::mp_set_contains<TypeList, std::remove_cv_t<T>>;
template <typename T, typename TypeList>
inline constexpr bool is_contained_v = is_contained<T, TypeList>::value;

// type list append
template <class... L> using tl_append = boost::mp11::mp_append<L...>;

// value_list
template <typename T, T... Values> struct value_list;

template <typename T, T Head, T... Tail> struct value_list<T, Head, Tail...> {
  static constexpr T head = Head;
  using tail = value_list<T, Tail...>;
};

template <typename T> struct value_list<T> {};

template <typename T> using tail_t = typename T::tail;

// is_contained_value
template <typename T, T Value, typename ValueList>
struct is_contained_value
    : std::conditional_t<Value == ValueList::head, std::true_type,
                         is_contained_value<T, Value, tail_t<ValueList>>> {};

template <typename T, T Value>
struct is_contained_value<T, Value, value_list<T>> : std::false_type {};

//  address_space_list
template <access::address_space... Values>
using address_space_list = value_list<access::address_space, Values...>;

template <access::address_space AddressSpace, typename ValueList>
using is_one_of_spaces =
    is_contained_value<access::address_space, AddressSpace, ValueList>;

// size type predicates
template <typename T1, typename T2>
struct is_type_size_equal : std::bool_constant<(sizeof(T1) == sizeof(T2))> {};

template <typename T1, typename T2>
struct is_type_size_double_of
    : std::bool_constant<(sizeof(T1) == (sizeof(T2) * 2))> {};

// find required type
template <typename TypeList, template <typename, typename> class Comp,
          typename T>
struct find_type {
  template <class T2> using C = Comp<T2, T>; // bind back
  using l = boost::mp11::mp_copy_if<TypeList, C>;
  using type = boost::mp11::mp_eval_if<is_empty_type_list<l>, void,
                                       boost::mp11::mp_front, l>;
};

template <typename TypeList, template <typename, typename> class Comp,
          typename T>
using find_type_t = typename find_type<TypeList, Comp, T>::type;

template <typename TypeList, typename T>
using find_same_size_type_t = find_type_t<TypeList, is_type_size_equal, T>;

template <typename TypeList, typename T>
using find_twice_as_large_type_t =
    find_type_t<TypeList, is_type_size_double_of, T>;

} // namespace detail
} // namespace _V1
} // namespace sycl
