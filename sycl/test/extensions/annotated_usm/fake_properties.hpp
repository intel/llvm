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

// Compile-time properties
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

inline constexpr bar_key::value_t bar;
template <int K> inline constexpr baz_key::value_t<K> baz;
template <typename... Ts> inline constexpr boo_key::value_t<Ts...> boo;

template <> struct is_property_key<bar_key> : std::true_type {};
template <> struct is_property_key<baz_key> : std::true_type {};
template <> struct is_property_key<boo_key> : std::true_type {};

template <typename objectT>
struct is_property_key_of<bar_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<baz_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<boo_key, objectT> : std::true_type {};

namespace detail {
template <> struct PropertyToKind<bar_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 0);
};
template <> struct PropertyToKind<baz_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 1);
};

template <> struct PropertyToKind<boo_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 2);
};

template <> struct IsCompileTimeProperty<bar_key> : std::true_type {};
template <> struct IsCompileTimeProperty<baz_key> : std::true_type {};
template <> struct IsCompileTimeProperty<boo_key> : std::true_type {};

} // namespace detail

// Runtime properties
enum foo_enum : unsigned { a, b, c };
struct foo_key {
  using value_t = property_value<foo_key>;
};
using foo = property_value<foo_key>;
template <> struct property_value<foo_key> {
  constexpr property_value(foo_enum v) : value(v) {}
  foo_enum value;
};

struct foz_key {
  using value_t = property_value<foz_key>;
};
using foz = property_value<foz_key>;
template <> struct property_value<foz_key> {
  float value1;
  bool value2;
};

#define rt_prop(N)                                                             \
  struct rt_prop##N##_key {                                                    \
    using value_t = property_value<rt_prop##N##_key>;                          \
  };                                                                           \
  using rt_prop##N = property_value<rt_prop##N##_key>;                         \
  template <> struct property_value<rt_prop##N##_key> {}

rt_prop(1);
rt_prop(2);
rt_prop(3);
rt_prop(4);
rt_prop(5);
rt_prop(6);
rt_prop(7);
rt_prop(8);
rt_prop(9);
rt_prop(10);
rt_prop(11);
rt_prop(12);
rt_prop(13);
rt_prop(14);
rt_prop(15);
rt_prop(16);
rt_prop(17);
rt_prop(18);
rt_prop(19);
rt_prop(20);
rt_prop(21);
rt_prop(22);
rt_prop(23);
rt_prop(24);
rt_prop(25);
rt_prop(26);
rt_prop(27);
rt_prop(28);
rt_prop(29);
rt_prop(30);
rt_prop(31);
rt_prop(32);
rt_prop(33);

#undef rt_prop

template <> struct is_property_key<foo_key> : std::true_type {};
template <> struct is_property_key<foz_key> : std::true_type {};

template <> struct is_property_key<rt_prop1_key> : std::true_type {};
template <> struct is_property_key<rt_prop2_key> : std::true_type {};
template <> struct is_property_key<rt_prop3_key> : std::true_type {};
template <> struct is_property_key<rt_prop4_key> : std::true_type {};
template <> struct is_property_key<rt_prop5_key> : std::true_type {};
template <> struct is_property_key<rt_prop6_key> : std::true_type {};
template <> struct is_property_key<rt_prop7_key> : std::true_type {};
template <> struct is_property_key<rt_prop8_key> : std::true_type {};
template <> struct is_property_key<rt_prop9_key> : std::true_type {};
template <> struct is_property_key<rt_prop10_key> : std::true_type {};
template <> struct is_property_key<rt_prop11_key> : std::true_type {};
template <> struct is_property_key<rt_prop12_key> : std::true_type {};
template <> struct is_property_key<rt_prop13_key> : std::true_type {};
template <> struct is_property_key<rt_prop14_key> : std::true_type {};
template <> struct is_property_key<rt_prop15_key> : std::true_type {};
template <> struct is_property_key<rt_prop16_key> : std::true_type {};
template <> struct is_property_key<rt_prop17_key> : std::true_type {};
template <> struct is_property_key<rt_prop18_key> : std::true_type {};
template <> struct is_property_key<rt_prop19_key> : std::true_type {};
template <> struct is_property_key<rt_prop20_key> : std::true_type {};
template <> struct is_property_key<rt_prop21_key> : std::true_type {};
template <> struct is_property_key<rt_prop22_key> : std::true_type {};
template <> struct is_property_key<rt_prop23_key> : std::true_type {};
template <> struct is_property_key<rt_prop24_key> : std::true_type {};
template <> struct is_property_key<rt_prop25_key> : std::true_type {};
template <> struct is_property_key<rt_prop26_key> : std::true_type {};
template <> struct is_property_key<rt_prop27_key> : std::true_type {};
template <> struct is_property_key<rt_prop28_key> : std::true_type {};
template <> struct is_property_key<rt_prop29_key> : std::true_type {};
template <> struct is_property_key<rt_prop30_key> : std::true_type {};
template <> struct is_property_key<rt_prop31_key> : std::true_type {};
template <> struct is_property_key<rt_prop32_key> : std::true_type {};
template <> struct is_property_key<rt_prop33_key> : std::true_type {};

template <typename objectT>
struct is_property_key_of<foo_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<foz_key, objectT> : std::true_type {};

template <typename objectT>
struct is_property_key_of<rt_prop1_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop2_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop3_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop4_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop5_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop6_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop7_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop8_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop9_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop10_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop11_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop12_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop13_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop14_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop15_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop16_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop17_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop18_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop19_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop20_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop21_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop22_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop23_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop24_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop25_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop26_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop27_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop28_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop29_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop30_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop31_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop32_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<rt_prop33_key, objectT> : std::true_type {};

namespace detail {
template <> struct PropertyToKind<foo_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 3);
};

template <> struct PropertyToKind<foz_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4);
};

template <> struct PropertyToKind<rt_prop1_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 1);
};
template <> struct PropertyToKind<rt_prop2_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 2);
};
template <> struct PropertyToKind<rt_prop3_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 3);
};
template <> struct PropertyToKind<rt_prop4_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 4);
};
template <> struct PropertyToKind<rt_prop5_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 5);
};
template <> struct PropertyToKind<rt_prop6_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 6);
};
template <> struct PropertyToKind<rt_prop7_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 7);
};
template <> struct PropertyToKind<rt_prop8_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 8);
};
template <> struct PropertyToKind<rt_prop9_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 9);
};
template <> struct PropertyToKind<rt_prop10_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 10);
};
template <> struct PropertyToKind<rt_prop11_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 11);
};
template <> struct PropertyToKind<rt_prop12_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 12);
};
template <> struct PropertyToKind<rt_prop13_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 13);
};
template <> struct PropertyToKind<rt_prop14_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 14);
};
template <> struct PropertyToKind<rt_prop15_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 15);
};
template <> struct PropertyToKind<rt_prop16_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 16);
};
template <> struct PropertyToKind<rt_prop17_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 17);
};
template <> struct PropertyToKind<rt_prop18_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 18);
};
template <> struct PropertyToKind<rt_prop19_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 19);
};
template <> struct PropertyToKind<rt_prop20_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 20);
};
template <> struct PropertyToKind<rt_prop21_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 21);
};
template <> struct PropertyToKind<rt_prop22_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 22);
};
template <> struct PropertyToKind<rt_prop23_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 23);
};
template <> struct PropertyToKind<rt_prop24_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 24);
};
template <> struct PropertyToKind<rt_prop25_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 25);
};
template <> struct PropertyToKind<rt_prop26_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 26);
};
template <> struct PropertyToKind<rt_prop27_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 27);
};
template <> struct PropertyToKind<rt_prop28_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 28);
};
template <> struct PropertyToKind<rt_prop29_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 29);
};
template <> struct PropertyToKind<rt_prop30_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 30);
};
template <> struct PropertyToKind<rt_prop31_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 31);
};
template <> struct PropertyToKind<rt_prop32_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 32);
};
template <> struct PropertyToKind<rt_prop33_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4 + 33);
};

template <> struct IsRuntimeProperty<foo_key> : std::true_type {};
template <> struct IsRuntimeProperty<foz_key> : std::true_type {};

template <> struct IsRuntimeProperty<rt_prop1_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop2_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop3_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop4_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop5_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop6_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop7_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop8_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop9_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop10_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop11_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop12_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop13_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop14_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop15_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop16_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop17_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop18_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop19_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop20_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop21_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop22_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop23_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop24_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop25_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop26_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop27_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop28_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop29_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop30_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop31_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop32_key> : std::true_type {};
template <> struct IsRuntimeProperty<rt_prop33_key> : std::true_type {};

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
