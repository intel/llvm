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

struct foo {
  foo(int v) : value(v) {}
  int value;
};

inline bool operator==(const foo &lhs, const foo &rhs) {
  return lhs.value == rhs.value;
}
inline bool operator!=(const foo &lhs, const foo &rhs) { return !(lhs == rhs); }

struct foz {
  foz(float v1, bool v2) : value1(v1), value2(v2) {}
  // Define copy constructor to make foz non-trivially copyable
  foz(const foz &f) {
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

namespace detail {
template <> struct PropertyToKind<bar> {
  static constexpr PropKind Kind =
      static_cast<enum PropKind>(PropKind::PropKindSize + 0);
};
template <> struct PropertyToKind<baz> {
  static constexpr PropKind Kind =
      static_cast<enum PropKind>(PropKind::PropKindSize + 1);
};
template <> struct PropertyToKind<foo> {
  static constexpr PropKind Kind =
      static_cast<enum PropKind>(PropKind::PropKindSize + 2);
};
template <> struct PropertyToKind<boo> {
  static constexpr PropKind Kind =
      static_cast<enum PropKind>(PropKind::PropKindSize + 3);
};
template <> struct PropertyToKind<foz> {
  static constexpr PropKind Kind =
      static_cast<enum PropKind>(PropKind::PropKindSize + 4);
};

template <> struct IsCompileTimeProperty<bar> : std::true_type {};
template <> struct IsCompileTimeProperty<baz> : std::true_type {};
template <> struct IsCompileTimeProperty<boo> : std::true_type {};

template <> struct IsRuntimeProperty<foo> : std::true_type {};
template <> struct IsRuntimeProperty<foz> : std::true_type {};
} // namespace detail

inline constexpr bar::value_t bar_v;
template <int K> inline constexpr baz::value_t<K> baz_v;
template <typename... Ts> inline constexpr boo::value_t<Ts...> boo_v;

} // namespace oneapi
} // namespace ext

template <> struct is_property<ext::oneapi::bar> : std::true_type {};
template <> struct is_property<ext::oneapi::baz> : std::true_type {};
template <> struct is_property<ext::oneapi::boo> : std::true_type {};
template <> struct is_property<ext::oneapi::foo> : std::true_type {};
template <> struct is_property<ext::oneapi::foz> : std::true_type {};

template <typename syclObjectT>
struct is_property_of<ext::oneapi::bar, syclObjectT> : std::true_type {};
template <typename syclObjectT>
struct is_property_of<ext::oneapi::baz, syclObjectT> : std::true_type {};
template <typename syclObjectT>
struct is_property_of<ext::oneapi::boo, syclObjectT> : std::true_type {};
template <typename syclObjectT>
struct is_property_of<ext::oneapi::foo, syclObjectT> : std::true_type {};
template <typename syclObjectT>
struct is_property_of<ext::oneapi::foz, syclObjectT> : std::true_type {};

} // namespace sycl
