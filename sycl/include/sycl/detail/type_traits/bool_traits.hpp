//==---- bool_traits.hpp - Narrow bool-detection and base-type traits ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Minimal subset of type traits for bool detection and vec base-type rewriting.
// Kept narrow so headers like group.hpp can include only this rather than the
// full type_traits.hpp (which transitively pulls in aliases.hpp, item.hpp, and
// generic_type_traits.hpp).
//
// type_traits.hpp includes this header and re-exports the same names, so
// including either is safe — there is only one definition of each trait.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/type_traits/vec_marray_traits.hpp>

#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace detail {

// is_scalar_bool
template <typename T>
struct is_scalar_bool
    : std::bool_constant<std::is_same_v<std::remove_cv_t<T>, bool>> {};

// is_vector_bool: true iff T is vec<bool, N>
template <typename T> struct is_vector_bool_impl : std::false_type {};
template <typename T, int N>
struct is_vector_bool_impl<vec<T, N>> : is_scalar_bool<T> {};

template <typename T> struct is_vector_bool : is_vector_bool_impl<T> {};

// is_bool: true iff T is bool or vec<bool, N>
template <typename T> struct is_bool_impl : is_scalar_bool<T> {};
template <typename T, int N>
struct is_bool_impl<vec<T, N>> : is_scalar_bool<T> {};

template <typename T> struct is_bool : is_bool_impl<std::remove_cv_t<T>> {};

// change_base_type_t: rewrite the element type of a vec<T,N> to B;
// leaves non-vec types unchanged.
template <typename T, typename B> struct change_base_type {
  using type = B;
};
template <typename T, int N, typename B> struct change_base_type<vec<T, N>, B> {
  using type = vec<B, N>;
};
template <typename T, typename B>
using change_base_type_t = typename change_base_type<T, B>::type;

} // namespace detail
} // namespace _V1
} // namespace sycl
