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
//     `sycl::ext::oneapi::experimental::detail::PropKind::PropKindSize`.
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
//  1. Add a new enumerator to `sycl::ext::oneapi::detail::PropKind`
//     representing the new property. Increment
//     `sycl::ext::oneapi::experimental::detail::PropKind::PropKindSize`
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

#include <iosfwd>      // for nullptr_t
#include <stdint.h>    // for uint32_t
#include <type_traits> // for false_type

#include <sycl/detail/defines_elementary.hpp>

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

// List of all properties.
enum PropKind : uint32_t {
  DeviceImageScope = 0,
  HostAccess = 1,
  WorkGroupSize = 2,
  WorkGroupSizeHint = 3,
  SubGroupSize = 4,
  DeviceHas = 5,
  Alignment = 6,
  CacheConfig = 7,
  UseRootSync = 8,
  RegisterAllocMode = 9,
  GRFSize = 10,
  GRFSizeAutomatic = 11,
  ESIMDL1CacheHint = 12,
  ESIMDL2CacheHint = 13,
  ESIMDL3CacheHint = 14,
  UsmKind = 15,
  CacheControlReadHint = 16,
  CacheControlReadAssertion = 17,
  CacheControlWrite = 18,
  BuildOptions = 19,
  BuildLog = 20,
  FloatingPointControls = 21,
  DataPlacement = 22,
  ContiguousMemory = 23,
  FullGroup = 24,
  Naive = 25,
  WorkGroupProgress = 26,
  SubGroupProgress = 27,
  WorkItemProgress = 28,
  NDRangeKernel = 29,
  SingleTaskKernel = 30,
  IndirectlyCallable = 31,
  CallsIndirectly = 32,
  InputDataPlacement = 33,
  OutputDataPlacement = 34,
  IncludeFiles = 35,
  RegisteredNames = 36,
  ClusterLaunch = 37,
  MaxWorkGroupSize = 38,
  MaxLinearWorkGroupSize = 39,
  Prefetch = 40,
  Deterministic = 41,
  InitializeToIdentity = 42,
  WorkGroupScratchSize = 43,
  Unaliased = 44,
  EventMode = 45,
  NativeLocalBlockIO = 46,
  InitialThreshold = 47,
  MaximumSize = 48,
  ZeroInit = 49,
  FastLink = 50,
  // PropKindSize must always be the last value.
  PropKindSize = 51,
};

template <typename PropertyT> struct PropertyToKind {
  static constexpr PropKind Kind = PropertyT::Kind;
};

struct property_tag {};

// This is used to implement has/get_property via inheritance queries.
template <typename property_key_t> struct property_key_tag : property_tag {};

template <typename property_t, PropKind Kind,
          typename property_key_t = property_t>
struct property_base : property_key_tag<property_key_t> {
  using key_t = property_key_t;

protected:
  constexpr property_t get_property_impl(property_key_tag<key_t>) const {
    return *static_cast<const property_t *>(this);
  }

  // For key_t access in error reporting specialization.
  template <typename> friend class __SYCL_EBO properties;

#if !defined(_MSC_VER)
  // Temporary, to ensure new code matches previous behavior and to catch any
  // silly copy-paste mistakes. MSVC can't compile it, but linux-only is
  // enough for this temporary check.
  static_assert([]() constexpr -> bool {
    if constexpr (std::is_same_v<property_t, key_t>)
      // key_t is incomplete at this point for runtime properties.
      return true;
    else
      return Kind == PropertyToKind<key_t>::Kind;
  }());
#endif
};

struct property_key_base_tag {};
struct compile_time_property_key_base_tag : property_key_base_tag {};

template <typename property_t, PropKind Kind_>
struct run_time_property_key : property_key_base_tag,
                               property_base<property_t, Kind_> {
protected:
  static constexpr PropKind Kind = Kind_;

  template <typename T>
  friend struct PropertyToKind;
};

template <PropKind Kind_>
struct compile_time_property_key : compile_time_property_key_base_tag {
protected:
  static constexpr PropKind Kind = Kind_;

  template <typename T>
  friend struct PropertyToKind;
};

// Get unique ID for property.
template <typename PropertyT> struct PropertyID {
  static constexpr int value =
      static_cast<int>(PropertyToKind<PropertyT>::Kind);
};

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
