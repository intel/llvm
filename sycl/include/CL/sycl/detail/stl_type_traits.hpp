//==------- stl_type_traits.hpp - SYCL STL type traits analogs -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines.hpp>
#include <iterator>
#include <memory>
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

// C++17
template <bool V> using bool_constant = std::integral_constant<bool, V>;

template <class...> using void_t = void;

// Custom type traits
template <typename T>
using allocator_value_type_t = typename std::allocator_traits<T>::value_type;

template <typename T>
using allocator_pointer_t = typename std::allocator_traits<T>::pointer;

template <typename T>
using iterator_category_t = typename std::iterator_traits<T>::iterator_category;

template <typename T>
using iterator_value_type_t = typename std::iterator_traits<T>::value_type;

template <typename T>
using iterator_pointer_t = typename std::iterator_traits<T>::pointer;

template <typename T>
using iterator_to_const_type_t =
    std::is_const<typename std::remove_pointer<iterator_pointer_t<T>>::type>;

// TODO Align with C++ named requirements: LegacyOutputIterator
// https://en.cppreference.com/w/cpp/named_req/OutputIterator
template <typename T>
using output_iterator_requirements =
    void_t<iterator_category_t<T>,
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
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
