//==------- properties.hpp --- sycl_khr_properties extension --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the sycl_khr_properties extension: the
// `sycl::khr::properties` compile-time property list container and the
// associated classification traits.
//
// Design notes:
//  * The list stores its properties as private base classes. Properties whose
//    values are all compile-time (empty types) are NOT stored -- they are
//    default-constructed on retrieval -- so an all-compile-time list is
//    zero-overhead and a mixed list only pays for its runtime members. This
//    keeps the list trivially copyable and cheap to compile (no std::tuple).
//  * The list does not canonicalize (sort) its element order. Per the
//    extension, two lists built from the same properties in a different order
//    may have different types; comparison operators are intentionally not
//    provided.
//
//===----------------------------------------------------------------------===//

#pragma once

#ifdef __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS

#include <sycl/detail/defines_elementary.hpp> // for __SYCL_EBO
#include <sycl/feature_test.hpp>              // for SYCL_KHR_PROPERTIES

#include <type_traits>

namespace sycl {
inline namespace _V1 {
// Forward declarations of the SYCL classes that the named properties below
// register against (via is_property_key_for). Only incomplete types are needed.
class queue;

namespace khr {

template <typename... EncodedProperties> class __SYCL_EBO properties;

// The property-definition infrastructure lives in `sycl::khr::property::detail`
// (not `sycl::khr::detail`) so it does not shadow `sycl::detail` for
// unqualified `detail::` lookups in sibling headers under `namespace
// sycl::khr`.
namespace property {
namespace detail {

//===----------------------------------------------------------------------===//
// Base tags
//
// Every property derives (directly or via a convenience base) from
// `property_tag`, and exposes a `__detail_key_t` alias naming its key. Every
// key derives from `property_key_tag`; a key whose property has no
// runtime-provided values additionally derives from
// `compile_time_property_key_tag`.
//===----------------------------------------------------------------------===//

struct property_tag {};
struct property_key_tag {};
struct compile_time_property_key_tag : property_key_tag {};

// Base for any property. `Key` is the property's associated key type.
template <typename Key> struct property_base : property_tag {
  using __detail_key_t = Key;
};

// True if `Prop` has at least one runtime-provided value and therefore needs to
// be stored in the list (runtime and hybrid properties). A property is
// storage-free exactly when its key is a compile-time key.
template <typename Prop>
inline constexpr bool __detail_has_runtime_value =
    !std::is_base_of_v<compile_time_property_key_tag,
                       typename Prop::__detail_key_t>;

//===----------------------------------------------------------------------===//
// Convenience base classes for defining properties
//
// These mirror the patterns in the extension's "Examples for implementors".
// Implementations may define properties directly on `property_base`/the tags,
// but these reduce boilerplate for the common shapes.
//===----------------------------------------------------------------------===//

// Base for a runtime property key (all of the property's values are supplied at
// runtime). Usage:
//   struct my_key : property::detail::runtime_property_key {};
//   struct my_prop : property::detail::runtime_property<my_key> { int value;
//   ... };
struct runtime_property_key : property_key_tag {};
template <typename Key> struct runtime_property : property_base<Key> {};

// Base for a compile-time property key with a single non-type value. Usage:
//   struct my_key : property::detail::constant_value_property_key {};
//   template <int V>
//   inline constexpr my_key::__detail_property_t<my_key, int, V> my_prop;
struct constant_value_property_key : compile_time_property_key_tag {
  template <typename PropertyKey, typename Type, Type Value>
  struct __detail_property_t : property_base<PropertyKey> {
    static constexpr Type value = Value;
  };
};

// Base for a compile-time property key with a single type value. Usage:
//   struct my_key : property::detail::constant_type_property_key {};
//   template <typename T>
//   inline constexpr my_key::__detail_property_t<my_key, T> my_prop;
struct constant_type_property_key : compile_time_property_key_tag {
  template <typename PropertyKey, typename Type>
  struct __detail_property_t : property_base<PropertyKey> {
    using value_t = Type;
  };
};

// Base for a hybrid property key (some values compile-time, some runtime). The
// key is a runtime key (the property carries runtime data and is stored).
// Usage:
//   struct my_key : property::detail::hybrid_property_key {};
//   template <int X> struct my_prop : property::detail::hybrid_property<my_key>
//   {
//     static constexpr int x = X; int y; constexpr my_prop(int y):y{y}{} };
struct hybrid_property_key : property_key_tag {};
template <typename Key> struct hybrid_property : property_base<Key> {};

//===----------------------------------------------------------------------===//
// Retrieval / storage machinery
//===----------------------------------------------------------------------===//

// Selects, from a pack of properties, the one whose key is `Key`. Assumes
// exactly one match (enforced by the "no duplicate key" mandate).
template <typename Key, typename... Properties> struct property_of_key;
template <typename Key, typename P, typename... Rest>
struct property_of_key<Key, P, Rest...>
    : std::conditional_t<std::is_same_v<typename P::__detail_key_t, Key>,
                         std::enable_if<true, P>,
                         property_of_key<Key, Rest...>> {};
template <typename Key, typename... Properties>
using property_of_key_t = typename property_of_key<Key, Properties...>::type;

// Returns (by const-ref) the argument whose key is `Key`.
template <typename Key, typename P, typename... Rest>
constexpr const auto &get_arg_by_key(const P &p, const Rest &...rest) {
  if constexpr (std::is_same_v<typename P::__detail_key_t, Key>)
    return p;
  else
    return get_arg_by_key<Key>(rest...);
}

// Storage base: inherits exactly the properties that have runtime values.
// Constructed from the full property pack; each stored base is initialized from
// the matching argument.
template <typename... Stored> struct property_storage : Stored... {
  constexpr property_storage() = default;
  template <typename... All>
  constexpr property_storage(const All &...all)
      : Stored(get_arg_by_key<typename Stored::__detail_key_t>(all...))... {}
};

// Builds `property_storage<subset>` where `subset` is the properties in `All`
// that have runtime values.
template <typename Selected, typename... Rest> struct build_storage;
template <typename... Sel> struct build_storage<property_storage<Sel...>> {
  using type = property_storage<Sel...>;
};
template <typename... Sel, typename P, typename... Rest>
struct build_storage<property_storage<Sel...>, P, Rest...>
    : build_storage<std::conditional_t<__detail_has_runtime_value<P>,
                                       property_storage<Sel..., P>,
                                       property_storage<Sel...>>,
                    Rest...> {};
template <typename... All>
using storage_for = typename build_storage<property_storage<>, All...>::type;

// True if T is a khr::properties list. Used to exclude the list itself from
// is_property (the list privately inherits its properties, so is_base_of would
// otherwise report it as a property).
template <typename> struct is_properties_list : std::false_type {};
template <typename... Ps>
struct is_properties_list<properties<Ps...>> : std::true_type {};

} // namespace detail
} // namespace property

//===----------------------------------------------------------------------===//
// Property traits
//===----------------------------------------------------------------------===//

template <typename T>
struct is_property
    : std::bool_constant<std::is_base_of_v<property::detail::property_tag, T> &&
                         !property::detail::is_properties_list<T>::value> {};
template <typename T>
inline constexpr bool is_property_v = is_property<T>::value;

template <typename T>
struct is_property_key
    : std::is_base_of<property::detail::property_key_tag, T> {};
template <typename T>
inline constexpr bool is_property_key_v = is_property_key<T>::value;

template <typename T>
struct is_property_key_compile_time
    : std::is_base_of<property::detail::compile_time_property_key_tag, T> {};
template <typename T>
inline constexpr bool is_property_key_compile_time_v =
    is_property_key_compile_time<T>::value;

// Customization point: a property opts in for a class by specializing this
// trait for the property's key and the supported class(es).
template <typename T, typename Class>
struct is_property_key_for : std::false_type {};
template <typename T, typename Class>
inline constexpr bool is_property_key_for_v =
    is_property_key_for<T, Class>::value;

// A property is for `Class` iff its key is for `Class`.
template <typename T, typename Class, typename = void>
struct is_property_for : std::false_type {};
template <typename T, typename Class>
struct is_property_for<T, Class, std::enable_if_t<is_property_v<T>>>
    : is_property_key_for<typename T::__detail_key_t, Class> {};
template <typename T, typename Class>
inline constexpr bool is_property_for_v = is_property_for<T, Class>::value;

// A property list all of whose properties are for `Class`. The empty list is
// for any class.
template <typename T, typename Class>
struct is_property_list_for : std::false_type {};
template <typename... Ps, typename Class>
struct is_property_list_for<properties<Ps...>, Class>
    : std::bool_constant<(is_property_for_v<Ps, Class> && ...)> {};
template <typename T, typename Class>
inline constexpr bool is_property_list_for_v =
    is_property_list_for<T, Class>::value;

//===----------------------------------------------------------------------===//
// The properties class
//===----------------------------------------------------------------------===//

template <typename... EncodedProperties>
class __SYCL_EBO properties
    : private property::detail::storage_for<EncodedProperties...> {
  using storage_t = property::detail::storage_for<EncodedProperties...>;

  static_assert((is_property_v<EncodedProperties> && ...),
                "Template arguments of khr::properties must be properties.");

  // Mandate: no two properties may share the same key.
  template <typename Key> static constexpr int key_count() {
    return (0 + ... +
            (std::is_same_v<typename EncodedProperties::__detail_key_t, Key>
                 ? 1
                 : 0));
  }
  static_assert(
      ((key_count<typename EncodedProperties::__detail_key_t>() == 1) && ...),
      "Duplicate properties in property list.");

public:
  template <typename... Properties>
  constexpr properties(Properties... props) : storage_t(props...) {}

  template <typename PropertyKey> static constexpr bool has_property() {
    return ((std::is_same_v<typename EncodedProperties::__detail_key_t,
                            PropertyKey>) ||
            ...);
  }

  // Compile-time key: the property carries no runtime value, so it is not
  // stored; return a default-constructed instance.
  template <typename PropertyKey>
  static constexpr auto get_property() -> std::enable_if_t<
      is_property_key_compile_time_v<PropertyKey>,
      property::detail::property_of_key_t<PropertyKey, EncodedProperties...>> {
    return property::detail::property_of_key_t<PropertyKey,
                                               EncodedProperties...>{};
  }

  // Runtime (or hybrid) key: return a copy of the stored property.
  //
  // The dummy `int` parameter gives this overload a parameter list distinct
  // from the static compile-time overload above. Without it, MSVC rejects the
  // two overloads (static vs non-static) as differing only in return type
  // (C2686) during class definition, before the enable_if SFINAE applies.
  template <typename PropertyKey>
  constexpr auto get_property(int = 0) const -> std::enable_if_t<
      !is_property_key_compile_time_v<PropertyKey>,
      property::detail::property_of_key_t<PropertyKey, EncodedProperties...>> {
    return static_cast<const property::detail::property_of_key_t<
        PropertyKey, EncodedProperties...> &>(*this);
  }
};

// Deduction guide. Per the extension, `EncodedProperties` need not equal
// `Properties`; today it does.
template <typename... Properties>
properties(Properties... props) -> properties<Properties...>;

using empty_properties_t = decltype(properties{});

//===----------------------------------------------------------------------===//
// Named properties
//
// The concrete properties defined by this extension live here (one header for
// the whole extension). Each property registers the classes it applies to via
// is_property_key_for. Constructor/overload support for these lives in the
// respective SYCL object headers (e.g. queue.hpp).
//===----------------------------------------------------------------------===//

namespace property {
namespace key {
struct enable_profiling : detail::runtime_property_key {};
struct in_order : detail::runtime_property_key {};
} // namespace key

// Queue properties.
struct enable_profiling : detail::runtime_property<key::enable_profiling> {
  constexpr enable_profiling(bool v = true) : value{v} {}
  bool value;
};
struct in_order : detail::runtime_property<key::in_order> {
  constexpr in_order(bool v = true) : value{v} {}
  bool value;
};
} // namespace property

template <>
struct is_property_key_for<property::key::enable_profiling, queue>
    : std::true_type {};
template <>
struct is_property_key_for<property::key::in_order, queue> : std::true_type {};

} // namespace khr
} // namespace _V1
} // namespace sycl

#endif // __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
