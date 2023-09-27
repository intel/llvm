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
namespace experimental {

struct bar_key {
  using value_t = property_value<bar_key>;
};

struct baz_key {
  template <int K>
  using value_t = property_value<baz_key, std::integral_constant<int, K>>;
};

struct boo_key {
  template <typename... Ts> using value_t = property_value<boo_key, Ts...>;
};

enum foo_enum : unsigned { a, b, c };
struct foo {
  constexpr foo(foo_enum v) : value(v) {}
  foo_enum value;
};

struct foz {
  float value1;
  bool value2;
};

inline constexpr bar_key::value_t bar;
template <int K> inline constexpr baz_key::value_t<K> baz;
template <typename... Ts> inline constexpr boo_key::value_t<Ts...> boo;

using foo_key = foo;
using foz_key = foz;

template <> struct is_property_key<bar_key> : std::true_type {};
template <> struct is_property_key<baz_key> : std::true_type {};
template <> struct is_property_key<boo_key> : std::true_type {};
template <> struct is_property_key<foo_key> : std::true_type {};
template <> struct is_property_key<foz_key> : std::true_type {};

template <typename syclObjectT>
struct is_property_key_of<bar_key, syclObjectT> : std::true_type {};
template <typename syclObjectT>
struct is_property_key_of<baz_key, syclObjectT> : std::true_type {};
template <typename syclObjectT>
struct is_property_key_of<boo_key, syclObjectT> : std::true_type {};
template <typename syclObjectT>
struct is_property_key_of<foo_key, syclObjectT> : std::true_type {};
template <typename syclObjectT>
struct is_property_key_of<foz_key, syclObjectT> : std::true_type {};

namespace detail {
template <> struct PropertyToKind<bar_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 0);
};
template <> struct PropertyToKind<baz_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 1);
};
template <> struct PropertyToKind<foo_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 2);
};
template <> struct PropertyToKind<boo_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 3);
};
template <> struct PropertyToKind<foz_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4);
};

template <> struct IsCompileTimeProperty<bar_key> : std::true_type {};
template <> struct IsCompileTimeProperty<baz_key> : std::true_type {};
template <> struct IsCompileTimeProperty<boo_key> : std::true_type {};

template <> struct IsRuntimeProperty<foo_key> : std::true_type {};
template <> struct IsRuntimeProperty<foz_key> : std::true_type {};

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
