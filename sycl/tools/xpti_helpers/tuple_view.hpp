//==---------- tuple_view.hpp - Tuple View ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>

#include <functional>
#include <tuple>
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace xpti_helpers {
/// A helper class to provide tuple-like access to a contiguous chunk of memory.
template <typename... Ts> struct tuple_view { unsigned char *data; };

template <size_t I, typename T> struct tuple_view_element;

template <size_t I, typename Head, typename... Rest>
struct tuple_view_element<I, tuple_view<Head, Rest...>>
    : tuple_view_element<I - 1, tuple_view<Rest...>> {};

template <typename Head, typename... Rest>
struct tuple_view_element<0, tuple_view<Head, Rest...>> {
  using type = Head;
};

template <size_t I, typename T> struct tuple_view_offset;

template <size_t I, typename... Ts>
struct tuple_view_offset<I, tuple_view<Ts...>> {
  static constexpr size_t value =
      sizeof(typename tuple_view_element<I - 1, tuple_view<Ts...>>::type) +
      tuple_view_offset<I - 1, tuple_view<Ts...>>::value;
};

template <typename... Ts> struct tuple_view_offset<0, tuple_view<Ts...>> {
  static constexpr size_t value = 0;
};

template <size_t I, typename... Ts>
typename tuple_view_element<I, tuple_view<Ts...>>::type
get(tuple_view<Ts...> &t) {
  return *reinterpret_cast<std::add_pointer_t<
      typename tuple_view_element<I, tuple_view<Ts...>>::type>>(
      t.data + tuple_view_offset<I, tuple_view<Ts...>>::value);
}

template <typename T> struct tuple_view_size {};

template <typename... Ts>
struct tuple_view_size<tuple_view<Ts...>>
    : std::integral_constant<size_t, sizeof...(Ts)> {};

template <typename F, typename Tuple, size_t... Is>
decltype(auto) apply_impl(F &&f, Tuple &&t, std::index_sequence<Is...>) {
  return std::invoke(std::forward<F>(f),
                     xpti_helpers::get<Is>(std::forward<Tuple>(t))...);
}

template <typename F, typename Tuple> decltype(auto) apply(F &&f, Tuple &&t) {
  return apply_impl(
      std::forward<F>(f), std::forward<Tuple>(t),
      std::make_index_sequence<
          tuple_view_size<std::remove_reference_t<Tuple>>::value>{});
}

template <typename R, typename T> struct as_function;

template <typename R, typename... Ts> struct as_function<R, std::tuple<Ts...>> {
  using type = std::function<R(Ts...)>;
};

template <typename T> struct as_tuple_view;

template <typename... Ts> struct as_tuple_view<std::tuple<Ts...>> {
  using type = tuple_view<Ts...>;
};

} // namespace xpti_helpers
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
