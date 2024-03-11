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

struct foo : detail::run_time_property_key<fakePropKind(3)> {
  constexpr foo(int v) : value(v) {}
  int value;
};

inline bool operator==(const foo &lhs, const foo &rhs) {
  return lhs.value == rhs.value;
}
inline bool operator!=(const foo &lhs, const foo &rhs) { return !(lhs == rhs); }

struct foz : detail::run_time_property_key<fakePropKind(4)> {
  constexpr foz(float v1, bool v2) : value1(v1), value2(v2) {}
  // Define copy constructor to make foz non-trivially copyable
  constexpr foz(const foz &f) {
    value1 = f.value1;
    value2 = f.value2;
  }
  float value1;
  bool value2;
};

inline bool operator==(const foz &lhs, const foz &rhs) {
  return lhs.value1 == rhs.value1 && lhs.value2 == rhs.value2;
}
inline bool operator!=(const foz &lhs, const foz &rhs) { return !(lhs == rhs); }

struct fir : detail::run_time_property_key<fakePropKind(5)> {
  // Intentionally not constexpr to test for properties that cannot be constexpr
  fir(float v1, bool v2) : value1(v1), value2(v2) {}
  // Define copy constructor to make foz non-trivially copyable
  fir(const foz &f) {
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

using foo_key = foo;
using foz_key = foz;
using fir_key = fir;

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
