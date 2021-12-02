//==---------- properties.hpp --- SYCL extension property tooling ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// HOW-TO: Add new compile-time property
//  1. Add a new enumerator to `sycl::ext::oneapi::detail::PropKind`
//     representing the new property. Increment
//     `sycl::ext::oneapi::detail::PropKind::PropKindSize`
//  2. Define property class with `value_t` that must be `property_value` with
//     the first template argument being the property class itself.
//  3. Add an `inline constexpr` variable in the same namespace as the property.
//     The variable should have the same type as `value_t` of the property class
//     and should be named as the property class with `_v` appended, e.g. for a
//     property `foo`, there should be a definition
//     `inline constexpr foo::value_t foo_v`.
//  4. Specialize `sycl::ext::oneapi::detail::PropertyToKind` for the new
//     property class. The specialization should have a `Kind` member with the
//     value equal to the enumerator added in 1.
//  5. Specialize `sycl::ext::oneapi::detail::IsCompileTimeProperty` for the new
//     property class. This specialization should derive from `std::true_type`.
//  6. Specialize `sycl::is_property` and `sycl::is_property_of` for the
//     property class.
/******************************** EXAMPLE **************************************
---------- sycl/include/sycl/ext/oneapi/property_list/properties.hpp -----------
// (1.)
enum PropKind : uint32_t {
  ...
  Bar,
  PropKindSize = N + 1, // N was the previous value
};
---------------------- path/to/new/property/file.hpp ---------------------------
namespace sycl {
namespace ext {
namespace oneapi {

// (2.)
struct bar {
  using value_t = property_value<bar>;
};

// (3.)
inline constexpr bar::value_t bar_v;

namespace detail {

// (4.)
template <> struct PropertyToKind<bar> {
  static constexpr PropKind Kind = PropKind::Bar;
};

// (5.)
template <> struct IsCompileTimeProperty<bar> : std::true_type {};

} // namespace detail
} // namespace oneapi
} // namespace ext

// (6.)
template <> struct is_property<ext::oneapi::bar> : std::true_type {};
// Replace SYCL_OBJ with the SYCL object to support the property.
template <> struct is_property_of<ext::oneapi::bar, SYCL_OBJ>
  : std::true_type {};

} // namespace sycl
*******************************************************************************/

// HOW-TO: Add new runtime property
//  1. Add a new enumerator to `sycl::ext::oneapi::detail::PropKind`
//     representing the new property. Increment
//     `sycl::ext::oneapi::detail::PropKind::PropKindSize`
//  2. Define property class.
//  3. Overload the `==` and `!=` operators for the new property class. The
//     comparison should compare all data members of the property class.
//  4. Specialize `sycl::ext::oneapi::detail::PropertyToKind` for the new
//     property class. The specialization should have a `Kind` member with the
//     value equal to the enumerator added in 1.
//  5. Specialize `sycl::ext::oneapi::detail::IsRuntimeProperty` for the new
//     property class. This specialization should derive from `std::true_type`.
//  6. Specialize `sycl::is_property` and `sycl::is_property_of` for the
//     property class.
/******************************* EXAMPLE ***************************************
---------- sycl/include/sycl/ext/oneapi/property_list/properties.hpp -----------
// (1.)
enum PropKind : uint32_t {
  ...
  Foo,
  PropKindSize = N + 1, // N was the previous value
};
---------------------- path/to/new/property/file.hpp ---------------------------
namespace sycl {
namespace ext {
namespace oneapi {

// (2.)
struct foo {
  foo(int v) : value(v) {}
  int value;
};

// (3.)
inline bool operator==(const foo &lhs, const foo &rhs) {
  return lhs.value == rhs.value;
}
inline bool operator!=(const foo &lhs, const foo &rhs) {
  return !(lhs == rhs);
}

namespace detail {

// (4.)
template <> struct PropertyToKind<foo> {
  static constexpr PropKind Kind = PropKind::Foo;
};

// (5.)
template <> struct IsRuntimeProperty<foo> : std::true_type {};
} // namespace detail
} // namespace oneapi
} // namespace ext

// (6.)
template <> struct is_property<ext::oneapi::foo> : std::true_type {};
// Replace SYCL_OBJ with the SYCL object to support the property.
template <> struct is_property_of<ext::oneapi::foo, SYCL_OBJ>
  : std::true_type {};

} // namespace sycl
*******************************************************************************/

#pragma once

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace detail {

// List of all properties.
enum PropKind : uint32_t {
  PropKindSize = 0,
};

// This trait must be specialized for all properties and must have a unique
// constexpr PropKind member named Kind.
template <typename PropertyT> struct PropertyToKind {};

// Get unique ID for property.
template <typename PropertyT> struct PropertyID {
  static constexpr int value =
      static_cast<int>(PropertyToKind<PropertyT>::Kind);
};

// Trait for identifying runtime properties.
template <typename PropertyT> struct IsRuntimeProperty : std::false_type {};

// Trait for identifying compile-time properties.
template <typename PropertyT> struct IsCompileTimeProperty : std::false_type {};

} // namespace detail
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
