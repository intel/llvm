//== tuple.hpp - limited trivially copy constructible implementation- C++ --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>

#include <cassert>
#include <iterator>
#include <tuple>
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

template <typename... T> struct tuple;

template <typename T, typename... Ts, std::size_t... Is>
std::tuple<Ts...> get_tuple_tail_impl(const std::tuple<T, Ts...> &Tuple,
                                      const std::index_sequence<Is...> &) {
  return std::tuple<Ts...>(std::get<Is + 1>(Tuple)...);
}

template <typename T, typename... Ts>
std::tuple<Ts...> get_tuple_tail(const std::tuple<T, Ts...> &Tuple) {
  return sycl::detail::get_tuple_tail_impl(
      Tuple, std::make_index_sequence<sizeof...(Ts)>());
}

template <typename... Ts> constexpr tuple<Ts...> make_tuple(Ts... Args) {
  return sycl::detail::tuple<Ts...>{Args...};
}

template <typename... Ts> auto tie(Ts &... Args) {
  return sycl::detail::tuple<Ts &...>(Args...);
}

template <std::size_t N, typename T> struct tuple_element;

template <std::size_t N, typename T, typename... Rest>
struct tuple_element<N, tuple<T, Rest...>>
    : tuple_element<N - 1, tuple<Rest...>> {};

template <typename T, typename... Rest>
struct tuple_element<0, tuple<T, Rest...>> {
  using type = T;
};

template <std::size_t I, class T>
using tuple_element_t = typename tuple_element<I, T>::type;

// Functor returning reference to the selected element of the tuple.
template <size_t N> struct get {
  template <typename... Ts>
  constexpr auto operator()(tuple<Ts...> &Tuple) const
      -> decltype(get<N - 1>()(Tuple.next)) {
    return get<N - 1>()(Tuple.next);
  }

  template <typename... Ts>
  constexpr auto operator()(const tuple<Ts...> &Tuple) const
      -> decltype(get<N - 1>()(Tuple.next)) {
    return get<N - 1>()(Tuple.next);
  }
};

// Functor returning reference to selected element of the tuple.
// Specialization stopping the recursion.
template <> struct get<0> {
  template <typename... Ts>
  using ret_type = typename tuple_element<0, tuple<Ts...>>::type;

  template <typename... Ts>
  constexpr ret_type<Ts...> &operator()(tuple<Ts...> &Tuple) const noexcept {
    return Tuple.holder.value;
  }

  template <typename... Ts>
  constexpr ret_type<Ts...> const &operator()(const tuple<Ts...> &Tuple) const
      noexcept {
    return Tuple.holder.value;
  }
};

template <typename T> struct TupleValueHolder {
  TupleValueHolder() = default;
  TupleValueHolder(const T &Value) : value(Value) {}
  T value;
};

// Tuple needs to be trivially_copy_assignable. Define operator= if necessary.
template <typename T,
          bool = std::is_trivially_copy_assignable<TupleValueHolder<T>>::value>
struct TupleCopyAssignableValueHolder : TupleValueHolder<T> {
  using TupleValueHolder<T>::TupleValueHolder;
};

template <typename T>
struct TupleCopyAssignableValueHolder<T, false> : TupleValueHolder<T> {
  using TupleValueHolder<T>::TupleValueHolder;

  TupleCopyAssignableValueHolder &
  operator=(const TupleCopyAssignableValueHolder &RHS) {
    this->value = RHS.value;
    return *this;
  }
};

template <typename T, typename... Ts> struct tuple<T, Ts...> {
  TupleCopyAssignableValueHolder<T> holder;
  tuple<Ts...> next;

  using tuple_type = std::tuple<T, Ts...>;

  tuple() = default;
  tuple(const tuple &) = default;
  template <typename UT, typename... UTs>
  tuple(const tuple<UT, UTs...> &RHS)
      : holder(RHS.holder.value), next(RHS.next) {}

  tuple(const T &Value, const Ts &... Next) : holder(Value), next(Next...) {}

  // required to convert std::tuple to inner tuple in user-provided functor
  tuple(const std::tuple<T, Ts...> &RHS)
      : holder(std::get<0>(RHS)), next(sycl::detail::get_tuple_tail(RHS)) {}

  // Convert to std::tuple with the same template arguments.
  operator std::tuple<T, Ts...>() const {
    return to_std_tuple(*this, std::make_index_sequence<sizeof...(Ts) + 1>());
  }

  // Convert to std::tuple with different template arguments.
  template <typename UT, typename... UTs>
  operator std::tuple<UT, UTs...>() const {
    return to_std_tuple(static_cast<tuple<UT, UTs...>>(*this),
                        std::make_index_sequence<sizeof...(Ts) + 1>());
  }

  template <typename UT, typename... UTs>
  tuple &operator=(const detail::tuple<UT, UTs...> &RHS) {
    holder.value = RHS.holder.value;
    next = RHS.next;
    return *this;
  }

  // if T is deduced with reference, compiler generates deleted operator= and,
  // since "template operator=" is not considered as operator= overload
  // the deleted operator= has a preference during lookup
  tuple &operator=(const detail::tuple<T, Ts...> &) = default;

  // Convert std::tuple to sycl::detail::tuple
  template <typename UT, typename... UTs>
  tuple &operator=(const std::tuple<UT, UTs...> &RHS) {
    holder.value = std::get<0>(RHS);
    next = sycl::detail::get_tuple_tail(RHS);
    return *this;
  }

  friend bool operator==(const tuple &LHS, const tuple &RHS) {
    return LHS.holder.value == RHS.holder.value && LHS.next == RHS.next;
  }
  friend bool operator!=(const tuple &LHS, const tuple &RHS) {
    return !(LHS == RHS);
  }

  template <typename UT, typename... UTs, std::size_t... Is>
  static std::tuple<UT, UTs...> to_std_tuple(const tuple<UT, UTs...> &Tuple,
                                             std::index_sequence<Is...>) {
    return std::tuple<UT, UTs...>(get<Is>()(Tuple)...);
  }
};

template <> struct tuple<> {
  using tuple_type = std::tuple<>;

  tuple() = default;
  tuple(const tuple &) = default;
  tuple(const std::tuple<> &) {}

  tuple &operator=(const tuple &) = default;
  tuple &operator=(const std::tuple<> &) { return *this; }
  friend bool operator==(const tuple &, const tuple &) { return true; }
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

namespace std {

template <size_t I, typename... Types>
constexpr typename tuple_element<I, tuple<Types...>>::type &
get(cl::sycl::detail::tuple<Types...> &Arg) noexcept {
  return cl::sycl::detail::get<I>()(Arg);
}

template <size_t I, typename... Types>
constexpr typename tuple_element<I, tuple<Types...>>::type const &
get(const cl::sycl::detail::tuple<Types...> &Arg) noexcept {
  return cl::sycl::detail::get<I>()(Arg);
}

} // namespace std
