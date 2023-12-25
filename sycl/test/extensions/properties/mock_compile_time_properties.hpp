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

#include <sycl/sycl.hpp>

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

struct foo_key {
  using value_t = property_value<foo_key>;
};
using foo = property_value<foo_key>;
template <> struct property_value<foo_key> {
  using key_t = foo_key;
  constexpr property_value(int v) : value(v) {}
  int value;
};

inline bool operator==(const foo &lhs, const foo &rhs) {
  return lhs.value == rhs.value;
}
inline bool operator!=(const foo &lhs, const foo &rhs) { return !(lhs == rhs); }

struct foz_key {
  using value_t = property_value<foz_key>;
};
using foz = property_value<foz_key>;
template <> struct property_value<foz_key> {
  using key_t = foz_key;
  constexpr property_value(float v1, bool v2) : value1(v1), value2(v2) {}
  // Define copy constructor to make foz non-trivially copyable
  constexpr property_value(const foz &f) : value1(f.value1), value2(f.value2) {}
  float value1;
  bool value2;
};

inline bool operator==(const foz &lhs, const foz &rhs) {
  return lhs.value1 == rhs.value1 && lhs.value2 == rhs.value2;
}
inline bool operator!=(const foz &lhs, const foz &rhs) { return !(lhs == rhs); }

struct fir_key {
  using value_t = property_value<fir_key>;
};
using fir = property_value<fir_key>;
template <> struct property_value<fir_key> {
  using key_t = fir_key;
  // Intentionally not constexpr to test for properties that cannot be constexpr
  property_value(float v1, bool v2) : value1(v1), value2(v2) {}
  // Define copy constructor to make foz non-trivially copyable
  property_value(const foz &f) {
    value1 = f.value1;
    value2 = f.value2;
  }
  float value1;
  bool value2;
};

inline bool operator==(const fir &lhs, const fir &rhs) {
  return lhs.value1 == rhs.value1 && lhs.value2 == rhs.value2;
}
inline bool operator!=(const fir &lhs, const fir &rhs) { return !(lhs == rhs); }

inline constexpr bar_key::value_t bar;
template <int K> inline constexpr baz_key::value_t<K> baz;
template <typename... Ts> inline constexpr boo_key::value_t<Ts...> boo;

template <> struct is_property_key<bar_key> : std::true_type {};
template <> struct is_property_key<baz_key> : std::true_type {};
template <> struct is_property_key<boo_key> : std::true_type {};
template <> struct is_property_key<foo_key> : std::true_type {};
template <> struct is_property_key<foz_key> : std::true_type {};
template <> struct is_property_key<fir_key> : std::true_type {};

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
template <typename syclObjectT>
struct is_property_key_of<fir_key, syclObjectT> : std::true_type {};

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
template <> struct PropertyToKind<fir_key> {
  static constexpr PropKind Kind = static_cast<enum PropKind>(
      static_cast<std::uint32_t>(PropKind::PropKindSize) + 4);
};

template <> struct IsCompileTimeProperty<bar_key> : std::true_type {};
template <> struct IsCompileTimeProperty<baz_key> : std::true_type {};
template <> struct IsCompileTimeProperty<boo_key> : std::true_type {};

template <> struct IsRuntimeProperty<foo_key> : std::true_type {};
template <> struct IsRuntimeProperty<foz_key> : std::true_type {};
template <> struct IsRuntimeProperty<fir_key> : std::true_type {};

template <typename Properties>
struct ConflictingProperties<boo_key, Properties>
    : ContainsProperty<fir_key, Properties> {};

template <typename Properties>
struct ConflictingProperties<fir_key, Properties>
    : ContainsProperty<boo_key, Properties> {};

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
