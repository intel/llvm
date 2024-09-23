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

inline constexpr detail::PropKind fakePropKind(int N) {
  return static_cast<detail::PropKind>(
      static_cast<std::uint32_t>(detail::PropKind::PropKindSize) + N);
}

// Compile-time properties
struct bar_key : detail::compile_time_property_key<fakePropKind(0)> {
  using value_t = property_value<bar_key>;
};

struct baz_key : detail::compile_time_property_key<fakePropKind(1)> {
  template <int K>
  using value_t = property_value<baz_key, std::integral_constant<int, K>>;
};

struct boo_key : detail::compile_time_property_key<fakePropKind(2)> {
  template <typename... Ts> using value_t = property_value<boo_key, Ts...>;
};

inline constexpr bar_key::value_t bar;
template <int K> inline constexpr baz_key::value_t<K> baz;
template <typename... Ts> inline constexpr boo_key::value_t<Ts...> boo;

template <typename objectT>
struct is_property_key_of<bar_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<baz_key, objectT> : std::true_type {};
template <typename objectT>
struct is_property_key_of<boo_key, objectT> : std::true_type {};

// Runtime properties
enum foo_enum : unsigned { a, b, c };
struct foo : detail::run_time_property_key<fakePropKind(3)> {
  constexpr foo(foo_enum v) : value(v) {}
  foo_enum value;
};

struct foz  : detail::run_time_property_key<fakePropKind(4)> {
  float value1;
  bool value2;

  foz(float value1, bool value2) : value1(value1), value2(value2) {}
};

struct rt_prop1 : detail::run_time_property_key<fakePropKind(5)> {};
struct rt_prop2 : detail::run_time_property_key<fakePropKind(6)> {};
struct rt_prop3 : detail::run_time_property_key<fakePropKind(7)> {};
struct rt_prop4 : detail::run_time_property_key<fakePropKind(8)> {};
struct rt_prop5 : detail::run_time_property_key<fakePropKind(9)> {};
struct rt_prop6 : detail::run_time_property_key<fakePropKind(10)> {};
struct rt_prop7 : detail::run_time_property_key<fakePropKind(11)> {};
struct rt_prop8 : detail::run_time_property_key<fakePropKind(12)> {};
struct rt_prop9 : detail::run_time_property_key<fakePropKind(13)> {};
struct rt_prop10 : detail::run_time_property_key<fakePropKind(14)> {};
struct rt_prop11 : detail::run_time_property_key<fakePropKind(15)> {};
struct rt_prop12 : detail::run_time_property_key<fakePropKind(16)> {};
struct rt_prop13 : detail::run_time_property_key<fakePropKind(17)> {};
struct rt_prop14 : detail::run_time_property_key<fakePropKind(18)> {};
struct rt_prop15 : detail::run_time_property_key<fakePropKind(19)> {};
struct rt_prop16 : detail::run_time_property_key<fakePropKind(20)> {};
struct rt_prop17 : detail::run_time_property_key<fakePropKind(21)> {};
struct rt_prop18 : detail::run_time_property_key<fakePropKind(22)> {};
struct rt_prop19 : detail::run_time_property_key<fakePropKind(23)> {};
struct rt_prop20 : detail::run_time_property_key<fakePropKind(24)> {};
struct rt_prop21 : detail::run_time_property_key<fakePropKind(25)> {};
struct rt_prop22 : detail::run_time_property_key<fakePropKind(26)> {};
struct rt_prop23 : detail::run_time_property_key<fakePropKind(27)> {};
struct rt_prop24 : detail::run_time_property_key<fakePropKind(28)> {};
struct rt_prop25 : detail::run_time_property_key<fakePropKind(29)> {};
struct rt_prop26 : detail::run_time_property_key<fakePropKind(30)> {};
struct rt_prop27 : detail::run_time_property_key<fakePropKind(31)> {};
struct rt_prop28 : detail::run_time_property_key<fakePropKind(32)> {};
struct rt_prop29 : detail::run_time_property_key<fakePropKind(33)> {};
struct rt_prop30 : detail::run_time_property_key<fakePropKind(34)> {};
struct rt_prop31 : detail::run_time_property_key<fakePropKind(35)> {};
struct rt_prop32 : detail::run_time_property_key<fakePropKind(36)> {};
struct rt_prop33 : detail::run_time_property_key<fakePropKind(37)> {};

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

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
