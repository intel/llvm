//==------- stl_type_traits.hpp - SYCL STL type traits analogs -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <iterator>    // for iterator_traits
#include <type_traits> // for is_const, remove_pointer_t, void_t
#include <utility>     // for declval

namespace sycl {
inline namespace _V1 {
namespace detail {

// Custom type traits.
// FIXME: Those doesn't seem to be a part of any published/future C++ standard
// so should probably be moved to a different place.
template <typename T>
using iterator_category_t = typename std::iterator_traits<T>::iterator_category;

template <typename T>
using iterator_value_type_t = typename std::iterator_traits<T>::value_type;

template <typename T>
using iterator_pointer_t = typename std::iterator_traits<T>::pointer;

template <typename T>
using iterator_to_const_type_t =
    std::is_const<std::remove_pointer_t<iterator_pointer_t<T>>>;

// TODO Align with C++ named requirements: LegacyOutputIterator
// https://en.cppreference.com/w/cpp/named_req/OutputIterator
template <typename T>
using output_iterator_requirements =
    std::void_t<iterator_category_t<T>,
                decltype(*std::declval<T>() =
                             std::declval<iterator_value_type_t<T>>())>;

template <typename, typename = void> struct is_output_iterator {
  static constexpr bool value = false;
};

template <typename T>
struct is_output_iterator<T, output_iterator_requirements<T>> {
  static constexpr bool value = true;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
