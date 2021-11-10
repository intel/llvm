//==--- mock_compile_time_properties.hpp -  Mock compile-time properties ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Since we do not currently have any compile-time properties this header cheats
// a little by defining its own compile-time properties. These are intended for
// testing and users should not replicate this behavior.

#pragma once

namespace sycl {
namespace ext {
namespace oneapi {

struct bar {
  using value_t = property_value<bar>;
};

struct baz {
  template <int K>
  using value_t = property_value<baz, std::integral_constant<int, K>>;
};

struct boo {
  template <typename... Ts> using value_t = property_value<boo, Ts...>;
};

namespace detail {
template <> struct CompileTimePropertyToKind<bar> {
  static constexpr CompileTimePropKind PropKind =
      static_cast<CompileTimePropKind>(
          CompileTimePropKind::CompileTimePropKindSize + 0);
};
template <> struct CompileTimePropertyToKind<baz> {
  static constexpr CompileTimePropKind PropKind =
      static_cast<CompileTimePropKind>(
          CompileTimePropKind::CompileTimePropKindSize + 1);
};
template <> struct CompileTimePropertyToKind<boo> {
  static constexpr CompileTimePropKind PropKind =
      static_cast<CompileTimePropKind>(
          CompileTimePropKind::CompileTimePropKindSize + 2);
};
} // namespace detail

inline constexpr bar::value_t bar_v;
template <int K> inline constexpr baz::value_t<K> baz_v;
template <typename... Ts> inline constexpr boo::value_t<Ts...> boo_v;

} // namespace oneapi
} // namespace ext

template <> struct is_property<ext::oneapi::bar> : std::true_type {};
template <> struct is_property<ext::oneapi::baz> : std::true_type {};
template <> struct is_property<ext::oneapi::boo> : std::true_type {};

template <typename syclObjectT>
struct is_property_of<ext::oneapi::bar, syclObjectT> : std::true_type {};
template <typename syclObjectT>
struct is_property_of<ext::oneapi::baz, syclObjectT> : std::true_type {};
template <typename syclObjectT>
struct is_property_of<ext::oneapi::boo, syclObjectT> : std::true_type {};

} // namespace sycl
