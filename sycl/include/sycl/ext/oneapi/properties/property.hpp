//==---------- properties.hpp --- SYCL extension property tooling ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// HOW-TO: Add new compile-time property
//  1. Add a new enumerator to `sycl::detail::PropKind` representing the new
//     property (this is the shared engine registry in
//     <sycl/detail/properties/property.hpp>). Increment
//     `sycl::detail::PropKind::PropKindSize`.
//  2. Define property key class inherited from
//     `detail::compile_time_property_key` with `value_t` that must be
//     `property_value` with the first template argument being the property
//     class itself. The name of the key class must be the property name
//     suffixed by `_key`, i.e. for a property `foo` the class should be named
//     `foo_key`.
//  3. Add an `inline constexpr` variable in the same namespace as the property
//     key. The variable should have the same type as `value_t` of the property
//     class, e.g. for a property `foo`, there should be a definition
//     `inline constexpr foo_key::value_t foo`.
//  4. Specialize `sycl::ext::oneapi::experimental::is_property_key_of` for the
//     property key class.
//  5. If the property needs an LLVM IR attribute, specialize
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
struct bar_key : detail::compile_time_property_key<PropKind::Bar> {
  using value_t = property_value<bar_key>;
};

// (3.)
inline constexpr bar_key::value_t bar;

// (4.)
// Replace SYCL_OBJ with the SYCL object to support the property.
template <> struct is_property_key_of<bar_key, SYCL_OBJ> : std::true_type {};

namespace detail {
// (5.)
template <> struct PropertyMetaInfo<bar_key::value_t> {
  static constexpr const char *name = "sycl-bar";
  static constexpr int value = 5;
};

} // namespace detail
} // namespace sycl::ext::oneapi::experimental
*******************************************************************************/

// HOW-TO: Add new runtime property
//  1. Add a new enumerator to `sycl::detail::PropKind` representing the new
//     property. Increment `sycl::detail::PropKind::PropKindSize`
//  2. Define property class, inheriting from `detail::run_time_property_key`.
//  3. Declare the property key as an alias to the property class. The name of
//     the key class must be the property name suffixed by `_key`, i.e. for a
//     property `foo` the class should be named `foo_key`.
//  4. Overload the `==` and `!=` operators for the new property class. The
//     comparison should compare all data members of the property class.
//  5. Specialize `sycl::ext::oneapi::experimental::is_property_key_of` for the
//     property class.
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
struct foo : detail::run_time_property_key<foo, PropKind::Foo> {
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
// Replace SYCL_OBJ with the SYCL object to support the property.
template <> struct is_property_key_of<foo, SYCL_OBJ> : std::true_type {};

} // namespace sycl::ext::oneapi::experimental
*******************************************************************************/

#pragma once

#include <cstddef>     // for nullptr_t
#include <type_traits> // for false_type

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/properties/property.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {
template <typename> class __SYCL_EBO properties;
// Property list traits
template <typename propertiesT> struct is_property_list : std::false_type {};
template <typename properties_list_ty>
struct is_property_list<properties<properties_list_ty>> : std::true_type {};
template <typename propertiesT>
inline constexpr bool is_property_list_v = is_property_list<propertiesT>::value;

namespace detail {

// The property-definition infrastructure has been promoted to the shared engine
// in <sycl/detail/properties/property.hpp> so it can be reused by the KHR
// properties layer. Re-export exactly those names here (rather than a blanket
// `using namespace sycl::detail;`) so existing experimental code that refers to
// them via `experimental::detail::...` keeps working, without pulling the rest
// of `sycl::detail` into this namespace's lookup.
using sycl::detail::compile_time_property_key;
using sycl::detail::compile_time_property_key_base_tag;
using sycl::detail::property_base;
using sycl::detail::property_key_base_tag;
using sycl::detail::property_key_tag;
using sycl::detail::property_tag;
using sycl::detail::PropertyID;
using sycl::detail::PropertyToKind;
using sycl::detail::PropKind;
using sycl::detail::run_time_property_key;

// Trait for property compile-time meta names and values.
template <typename PropertyT> struct PropertyMetaInfo {
  // Some properties don't have meaningful compile-time values.
  // Default to empty, as those will be ignored anyway.
  static constexpr const char *name = "";
  static constexpr std::nullptr_t value = nullptr;
};

template <typename> struct HasCompileTimeEffect : std::false_type {};

} // namespace detail

template <typename, typename> struct is_property_key_of : std::false_type {};

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
