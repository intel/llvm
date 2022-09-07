//==----------- type_list.hpp - SYCL list of types utils -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>
#include <sycl/detail/stl_type_traits.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
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

template <typename Head, typename... Tail> struct type_list<Head, Tail...> {
  using head = Head;
  using tail = type_list<Tail...>;
};

template <typename Head, typename... Tail, typename... Tail2>
struct type_list<type_list<Head, Tail...>, Tail2...> {
private:
  using remainder = tail_t<type_list<Head>>;
  static constexpr bool has_remainder = !is_empty_type_list<remainder>::value;
  using without_remainder = type_list<Tail..., Tail2...>;
  using with_remainder = type_list<remainder, Tail..., Tail2...>;

public:
  using head = head_t<type_list<Head>>;
  using tail = conditional_t<has_remainder, with_remainder, without_remainder>;
};

// is_contained
template <typename T, typename TypeList>
struct is_contained
    : conditional_t<std::is_same<remove_cv_t<T>, head_t<TypeList>>::value,
                    std::true_type, is_contained<T, tail_t<TypeList>>> {};

template <typename T>
struct is_contained<T, empty_type_list> : std::false_type {};

// value_list
template <typename T, T... Values> struct value_list;

template <typename T, T Head, T... Tail> struct value_list<T, Head, Tail...> {
  static constexpr T head = Head;
  using tail = value_list<T, Tail...>;
};

template <typename T> struct value_list<T> {};

// is_contained_value
template <typename T, T Value, typename ValueList>
struct is_contained_value
    : conditional_t<Value == ValueList::head, std::true_type,
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
template <typename TypeList, template <typename, typename> class Comp,
          typename T>
struct find_type {
  using head = head_t<TypeList>;
  using tail = typename find_type<tail_t<TypeList>, Comp, T>::type;
  using type = conditional_t<Comp<head, T>::value, head, tail>;
};

template <template <typename, typename> class Comp, typename T>
struct find_type<empty_type_list, Comp, T> {
  using type = void;
};

template <typename TypeList, template <typename, typename> class Comp,
          typename T>
using find_type_t = typename find_type<TypeList, Comp, T>::type;

template <typename TypeList, typename T>
using find_same_size_type_t = find_type_t<TypeList, is_type_size_equal, T>;

template <typename TypeList, typename T>
using find_smaller_type_t = find_type_t<TypeList, is_type_size_less, T>;

template <typename TypeList, typename T>
using find_larger_type_t = find_type_t<TypeList, is_type_size_greater, T>;

template <typename TypeList, typename T>
using find_twice_as_small_type_t =
    find_type_t<TypeList, is_type_size_half_of, T>;

template <typename TypeList, typename T>
using find_twice_as_large_type_t =
    find_type_t<TypeList, is_type_size_double_of, T>;

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
