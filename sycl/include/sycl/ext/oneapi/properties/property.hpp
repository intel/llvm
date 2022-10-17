//==---------- properties.hpp --- SYCL extension property tooling ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// HOW-TO: Add new compile-time property
//  1. Add a new enumerator to
//     `sycl::ext::oneapi::experimental::detail::PropKind` representing the new
//     property. Increment
//     `sycl::ext::oneapi::experimental::detail::PropKind::PropKindSize`
//  2. Define property key class with `value_t` that must be `property_value`
//     with the first template argument being the property class itself. The
//     name of the key class must be the property name suffixed by `_key`, i.e.
//     for a property `foo` the class should be named `foo_key`.
//  3. Add an `inline constexpr` variable in the same namespace as the property
//     key. The variable should have the same type as `value_t` of the property
//     class, e.g. for a property `foo`, there should be a definition
//     `inline constexpr foo_key::value_t foo`.
//  4. Specialize `sycl::ext::oneapi::experimental::is_property_key` and
//     `sycl::ext::oneapi::experimental::is_property_key_of` for the property
//     key class.
//  5. Specialize `sycl::ext::oneapi::experimental::detail::PropertyToKind` for
//     the new property key class. The specialization should have a `Kind`
//     member with the value equal to the enumerator added in 1.
//  6. Specialize
//     `sycl::ext::oneapi::experimental::detail::IsCompileTimeProperty` for the
//     new property key class. This specialization should derive from
//     `std::true_type`.
//  7. If the property needs an LLVM IR attribute, specialize
//     `sycl::ext::oneapi::experimental::detail::PropertyMetaInfo` for the new
//     `value_t` of the property key class. The specialization must have a
//     `static constexpr const char *name` member with a value equal to the
//     expected LLVM IR attribute name. The common naming scheme for these is
//     the name of the property with "_" replaced with "-" and "sycl-" appended,
//     for example a property `foo_bar` would have an LLVM IR attribute name
//     "sycl-foo-bar". Likewise, the specialization must have a `static
//     constexpr T value` member where `T` is either an integer, a floating
//     point, a boolean, an enum, a char, or a `const char *`, or a
//     `std::nullptr_t`. This will be the value of the generated LLVM IR
//     attribute. If `std::nullptr_t` is used the attribute will not have a
//     value.
/******************************** EXAMPLE **************************************
------------- sycl/include/sycl/ext/oneapi/properties/property.hpp -------------
// (1.)
enum PropKind : uint32_t {
  ...
  Bar,
  PropKindSize = N + 1, // N was the previous value
};
---------------------- path/to/new/property/file.hpp ---------------------------
namespace sycl::ext::oneapi::experimental {

// (2.)
struct bar_key {
  using value_t = property_value<bar_key>;
};

// (3.)
inline constexpr bar_key::value_t bar;

// (4.)
template <> struct is_property_key<bar_key> : std::true_type {};
// Replace SYCL_OBJ with the SYCL object to support the property.
template <> struct is_property_key_of<bar_key, SYCL_OBJ> : std::true_type {};

namespace detail {

// (5.)
template <> struct PropertyToKind<bar_key> {
  static constexpr PropKind Kind = PropKind::Bar;
};

// (6.)
template <> struct IsCompileTimeProperty<bar_key> : std::true_type {};

// (7.)
template <> struct PropertyMetaInfo<bar_key::value_t> {
  static constexpr const char *value = "sycl-bar";
  static constexpr int value = 5;
};

} // namespace detail
} // namespace sycl::ext::oneapi::experimental
*******************************************************************************/

// HOW-TO: Add new runtime property
//  1. Add a new enumerator to `sycl::ext::oneapi::detail::PropKind`
//     representing the new property. Increment
//     `sycl::ext::oneapi::experimental::detail::PropKind::PropKindSize`
//  2. Define property class.
//  3. Declare the property key as an alias to the property class. The name of
//     the key class must be the property name suffixed by `_key`, i.e. for a
//     property `foo` the class should be named `foo_key`.
//  4. Overload the `==` and `!=` operators for the new property class. The
//     comparison should compare all data members of the property class.
//  5. Specialize `sycl::ext::oneapi::experimental::is_property_key` and
//     `sycl::ext::oneapi::experimental::is_property_key_of` for the property
//     class.
//  6. Specialize `sycl::ext::oneapi::detail::PropertyToKind` for the new
//     property class. The specialization should have a `Kind` member with the
//     value equal to the enumerator added in 1.
//  7. Specialize `sycl::ext::oneapi::experimental::detail::IsRuntimeProperty`
//     for the new property class. This specialization should derive from
//     `std::true_type`.
/******************************* EXAMPLE ***************************************
------------- sycl/include/sycl/ext/oneapi/properties/property.hpp -------------
// (1.)
enum PropKind : uint32_t {
  ...
  Foo,
  PropKindSize = N + 1, // N was the previous value
};
---------------------- path/to/new/property/file.hpp ---------------------------
namespace sycl::ext::oneapi::experimental {

// (2.)
struct foo {
  foo(int v) : value(v) {}
  int value;
};

// 3.
using foo_key = foo;

// (4.)
inline bool operator==(const foo &lhs, const foo &rhs) {
  return lhs.value == rhs.value;
}
inline bool operator!=(const foo &lhs, const foo &rhs) {
  return !(lhs == rhs);
}

// (5.)
template <> struct is_property_key<foo> : std::true_type {};
// Replace SYCL_OBJ with the SYCL object to support the property.
template <> struct is_property_key_of<foo, SYCL_OBJ> : std::true_type {};

namespace detail {

// (6.)
template <> struct PropertyToKind<foo> {
  static constexpr PropKind Kind = PropKind::Foo;
};

// (7.)
template <> struct IsRuntimeProperty<foo> : std::true_type {};

} // namespace detail
} // namespace sycl::ext::oneapi::experimental
*******************************************************************************/

#pragma once

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {

// List of all properties.
enum PropKind : uint32_t {
  DeviceImageScope = 0,
  HostAccess = 1,
  InitMode = 2,
  ImplementInCSR = 3,
  LatencyAnchorID = 4,
  LatencyConstraint = 5,
  WorkGroupSize = 6,
  WorkGroupSizeHint = 7,
  SubGroupSize = 8,
  // PropKindSize must always be the last value.
  PropKindSize = 9,
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

// Trait for property compile-time meta names and values.
template <typename PropertyT> struct PropertyMetaInfo {
  // Some properties don't have meaningful compile-time values.
  // Default to empty, as those will be ignored anyway.
  static constexpr const char *name = "";
  static constexpr std::nullptr_t value = nullptr;
};

} // namespace detail

template <typename> struct is_property_key : std::false_type {};
template <typename, typename> struct is_property_key_of : std::false_type {};

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
