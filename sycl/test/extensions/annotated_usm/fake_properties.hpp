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
struct foo {
  constexpr foo(foo_enum v) : value(v) {}
  foo_enum value;
};

struct foz {
  float value1;
  bool value2;
};

struct rt_prop1 {};
struct rt_prop2 {};
struct rt_prop3 {};
struct rt_prop4 {};
struct rt_prop5 {};
struct rt_prop6 {};
struct rt_prop7 {};
struct rt_prop8 {};
struct rt_prop9 {};
struct rt_prop10 {};
struct rt_prop11 {};
struct rt_prop12 {};
struct rt_prop13 {};
struct rt_prop14 {};
struct rt_prop15 {};
struct rt_prop16 {};
struct rt_prop17 {};
struct rt_prop18 {};
struct rt_prop19 {};
struct rt_prop20 {};
struct rt_prop21 {};
struct rt_prop22 {};
struct rt_prop23 {};
struct rt_prop24 {};
struct rt_prop25 {};
struct rt_prop26 {};
struct rt_prop27 {};
struct rt_prop28 {};
struct rt_prop29 {};
struct rt_prop30 {};
struct rt_prop31 {};
struct rt_prop32 {};
struct rt_prop33 {};

using foo_key = foo;
using foz_key = foz;
using rt_prop1_key = rt_prop1;
using rt_prop2_key = rt_prop2;
using rt_prop3_key = rt_prop3;
using rt_prop4_key = rt_prop4;
using rt_prop5_key = rt_prop5;
using rt_prop6_key = rt_prop6;
using rt_prop7_key = rt_prop7;
using rt_prop8_key = rt_prop8;
using rt_prop9_key = rt_prop9;
using rt_prop10_key = rt_prop10;
using rt_prop11_key = rt_prop11;
using rt_prop12_key = rt_prop12;
using rt_prop13_key = rt_prop13;
using rt_prop14_key = rt_prop14;
using rt_prop15_key = rt_prop15;
using rt_prop16_key = rt_prop16;
using rt_prop17_key = rt_prop17;
using rt_prop18_key = rt_prop18;
using rt_prop19_key = rt_prop19;
using rt_prop20_key = rt_prop20;
using rt_prop21_key = rt_prop21;
using rt_prop22_key = rt_prop22;
using rt_prop23_key = rt_prop23;
using rt_prop24_key = rt_prop24;
using rt_prop25_key = rt_prop25;
using rt_prop26_key = rt_prop26;
using rt_prop27_key = rt_prop27;
using rt_prop28_key = rt_prop28;
using rt_prop29_key = rt_prop29;
using rt_prop30_key = rt_prop30;
using rt_prop31_key = rt_prop31;
using rt_prop32_key = rt_prop32;
using rt_prop33_key = rt_prop33;

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
