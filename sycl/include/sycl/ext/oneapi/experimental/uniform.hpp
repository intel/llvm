//==------ uniform.hpp - SYCL uniform extension --------*- C++ -*-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
// Implemenation of the sycl_ext_oneapi_uniform extension.
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_uniform.asciidoc
// ===--------------------------------------------------------------------=== //

#pragma once

// SYCL extension macro definition as required by the SYCL specification.
// 1 - Initial extension version. Base features are supported.
#define SYCL_EXT_ONEAPI_UNIFORM 1

#include <type_traits>

// Forward declarations of types not allowed to be wrapped in uniform:
namespace sycl {
inline namespace _V1 {

struct sub_group;
template <int, bool> class item;
template <int> class id;
template <int> class nd_item;
template <int> class h_item;
template <int> class group;
template <int> class nd_range;
using sycl::sub_group;

namespace ext::oneapi::experimental {
namespace detail {

template <class T, template <int> class Tmpl>
struct is_instance_of_tmpl_int : std::false_type {};
template <int N, template <int> class T, template <int> class Tmpl>
struct is_instance_of_tmpl_int<T<N>, Tmpl>
    : std::conditional<std::is_same_v<T<N>, Tmpl<N>>, std::true_type,
                       std::false_type> {};
template <class T, template <int> class Tmpl>
static inline constexpr bool is_instance_of_tmpl_int_v =
    is_instance_of_tmpl_int<T, Tmpl>::value;

template <class T, template <int, bool> class Tmpl>
struct is_instance_of_tmpl_int_bool : std::false_type {};
template <int N, bool X, template <int, bool> class T,
          template <int, bool> class Tmpl>
struct is_instance_of_tmpl_int_bool<T<N, X>, Tmpl>
    : std::conditional<std::is_same_v<T<N, X>, Tmpl<N, X>>, std::true_type,
                       std::false_type> {};
template <class T, template <int, bool> class Tmpl>
static inline constexpr bool is_instance_of_tmpl_int_bool_v =
    is_instance_of_tmpl_int_bool<T, Tmpl>::value;
} // namespace detail

template <class T> class uniform {
  template <class U> static constexpr bool can_be_uniform() {
    return !detail::is_instance_of_tmpl_int_bool_v<U, sycl::item> &&
           !detail::is_instance_of_tmpl_int_v<U, sycl::nd_item> &&
           !detail::is_instance_of_tmpl_int_v<U, sycl::h_item> &&
           !detail::is_instance_of_tmpl_int_v<U, sycl::group> &&
           !detail::is_instance_of_tmpl_int_v<U, sycl::nd_range> &&
           !std::is_same_v<U, sycl::sub_group>;
  }
  static_assert(can_be_uniform<T>() && "type not allowed to be `uniform`");

public:
  explicit uniform(T x) noexcept : Val(x) {}

  // TODO provide a ways to reflect this conversion from uniform to T in the IR
  // so that the compiler can take advantage of uniformness. Could be marked
  // with some intrinsic call like `__builtin_uniform_unwrap(Val);`

  /// Implicit conversion to the underlying type.
  operator const T() const { return Val; }

  uniform &operator=(const uniform &) = delete;

  /* Other explicitly deleted operators improve error messages
     if a user incorrectly attempts to modify a uniform */
  uniform &operator+=(const T &) = delete;
  uniform &operator-=(const T &) = delete;
  uniform &operator*=(const T &) = delete;
  uniform &operator/=(const T &) = delete;
  uniform &operator%=(const T &) = delete;
  uniform &operator&=(const T &) = delete;
  uniform &operator|=(const T &) = delete;
  uniform &operator^=(const T &) = delete;
  uniform &operator<<=(const T &) = delete;
  uniform &operator>>=(const T &) = delete;
  uniform &operator++() = delete;
  uniform &operator++(int) = delete;
  uniform &operator--() = delete;
  uniform &operator--(int) = delete;

private:
  T Val;
};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
