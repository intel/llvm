//==------------- types_elementary.hpp - DPC++ Explicit SIMD API -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Definitions and declarations which are used in ESIMD type system, and which
// don't depend on other ESIMD types.
//===----------------------------------------------------------------------===//

#pragma once

// ESIMD HEADERS SHOULD NOT BE INCLUDED HERE
#include <cstdint>
#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace ext::intel::esimd {

/// @cond ESIMD_DETAIL

namespace detail {

// Unsigned integer type of given byte size or 'void'.
template <int N>
using uint_type_t = std::conditional_t<
    N == 1, uint8_t,
    std::conditional_t<
        N == 2, uint16_t,
        std::conditional_t<N == 4, uint32_t,
                           std::conditional_t<N == 8, uint64_t, void>>>>;

template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

/// Base case for checking if a type U is one of the types.
template <typename U> constexpr bool is_type() { return false; }

template <typename U, typename T, typename... Ts> constexpr bool is_type() {
  using UU = typename std::remove_const_t<U>;
  using TT = typename std::remove_const_t<T>;
  return std::is_same_v<UU, TT> || is_type<UU, Ts...>();
}

// Converts types to single 'void' type (used for SFINAE).
template <class...> struct make_esimd_void {
  using type = void;
};
template <class... Tys>
using __esimd_void_t = typename make_esimd_void<Tys...>::type;

// Checks if standard arithmetic operations can be applied to given type.
template <class Ty, class = void>
struct is_esimd_arithmetic_type : std::false_type {};

template <class Ty>
struct is_esimd_arithmetic_type<
    Ty, __esimd_void_t<std::enable_if_t<std::is_arithmetic_v<Ty>>,
                       decltype(std::declval<Ty>() + std::declval<Ty>()),
                       decltype(std::declval<Ty>() - std::declval<Ty>()),
                       decltype(std::declval<Ty>() * std::declval<Ty>()),
                       decltype(std::declval<Ty>() / std::declval<Ty>())>>
    : std::true_type {};

template <typename Ty>
static inline constexpr bool is_esimd_arithmetic_type_v =
    is_esimd_arithmetic_type<Ty>::value;

// Checks if given type can serve as clang vector element type.
template <typename Ty>
struct is_vectorizable : std::conditional_t<is_esimd_arithmetic_type_v<Ty>,
                                            std::true_type, std::false_type> {};

template <typename Ty>
static inline constexpr bool is_vectorizable_v = is_vectorizable<Ty>::value;

// Raw vector type, using clang vector type extension.
template <typename Ty, int N> struct raw_vector_type {
  static_assert(!std::is_const_v<Ty>, "const element type not supported");
  static_assert(is_vectorizable_v<Ty>, "element type not supported");
  static_assert(N > 0, "zero-element vector not supported");

  static constexpr int length = N;
  using type = Ty __attribute__((ext_vector_type(N)));
};

// Alias for clang vector type with given element type and number of elements.
template <typename Ty, int N>
using vector_type_t = typename raw_vector_type<Ty, N>::type;

// Checks if given type T is a raw clang vector type, plus provides some info
// about it if it is.

struct invalid_element_type;

template <class T> struct is_clang_vector_type : std::false_type {
  static constexpr int length = 0;
  using element_type = invalid_element_type;
};

template <class T, int N>
struct is_clang_vector_type<T __attribute__((ext_vector_type(N)))>
    : std::true_type {
  static constexpr int length = N;
  using element_type = T;
};
template <class T>
static inline constexpr bool is_clang_vector_type_v =
    is_clang_vector_type<T>::value;

template <class T> struct vector_element_type;

template <class T, int N> struct vector_element_type<vector_type_t<T, N>> {
  using type = T;
};

template <class T, int N>
struct vector_element_type<T __attribute__((ext_vector_type(N)))> {
  using type = T;
};

template <class T>
using vector_element_type_t = typename vector_element_type<T>::type;

} // namespace detail

/// @endcond ESIMD_DETAIL

} // namespace ext::intel::esimd
} // namespace _V1
} // namespace sycl
