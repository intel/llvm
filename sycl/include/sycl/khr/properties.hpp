//==------- properties.hpp --- sycl_khr_properties extension --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the sycl_khr_properties extension: the
// `sycl::khr::properties` container and its classification traits.
//
// Compile-time-only properties are not stored (they are default-constructed on
// retrieval); the list preserves element order (no sorting) per the extension.
//
//===----------------------------------------------------------------------===//

#pragma once

#ifdef __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS

#include <sycl/detail/defines_elementary.hpp> // for __SYCL_EBO
#include <sycl/feature_test.hpp>              // for SYCL_KHR_PROPERTIES

#include <type_traits>

namespace sycl {
inline namespace _V1 {

namespace khr {

template <typename... EncodedProperties> class __SYCL_EBO properties;

// The property-definition infrastructure lives in `sycl::khr::property::detail`
// (not `sycl::khr::detail`) so it does not shadow `sycl::detail` for
// unqualified `detail::` lookups in sibling headers under `namespace
// sycl::khr`.
namespace property {
namespace detail {

// Every key derives from `property_key_base<Self, CompileTime>` and every
// property from `property_base<Self, Key>`. Both name the exact type `Self`, so
// the traits do not classify derived classes or property lists as keys or
// properties.
template <typename Self, bool CompileTime> struct property_key_base {
  using __detail_key_self_t = Self;
  static constexpr bool __detail_is_compile_time = CompileTime;
};

template <typename Self, typename Key> struct property_base {
  using __detail_property_self_t = Self;
  using __detail_key_t = Key;
};

// True if `Prop` has at least one runtime-provided value and therefore needs to
// be stored in the list (runtime and hybrid properties). A property is
// storage-free exactly when its key is a compile-time key.
template <typename Prop>
inline constexpr bool __detail_has_runtime_value =
    !Prop::__detail_key_t::__detail_is_compile_time;

// Convenience base classes for defining properties, mirroring the extension's
// examples for implementors. Properties may also be defined directly on the
// bases above.

// Runtime property key: all of the property's values are supplied at runtime.
template <typename Self>
struct runtime_property_key : property_key_base<Self, false> {};
template <typename Self, typename Key>
struct runtime_property : property_base<Self, Key> {};

// Compile-time key with a single non-type value.
template <typename Self>
struct constant_value_property_key : property_key_base<Self, true> {
  template <typename Type, Type Value>
  struct __detail_property_t
      : property_base<__detail_property_t<Type, Value>, Self> {
    static constexpr Type value = Value;
  };
};

// Compile-time key with a single type value.
template <typename Self>
struct constant_type_property_key : property_key_base<Self, true> {
  template <typename Type>
  struct __detail_property_t : property_base<__detail_property_t<Type>, Self> {
    using value_t = Type;
  };
};

// Hybrid property key: some values compile-time, some runtime. The property
// carries runtime data and is stored.
template <typename Self>
struct hybrid_property_key : property_key_base<Self, false> {};
template <typename Self, typename Key>
struct hybrid_property : property_base<Self, Key> {};

// Retrieval / storage machinery.

// Selects, from a pack of properties, the one whose key is `Key`. Assumes at
// most one match (enforced by the "no duplicate key" mandate); without a match
// there is no `type`, so `property_of_key_t` is a substitution failure.
template <typename Key, typename... Properties> struct property_of_key {};
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
// Exact-type checks for the self-naming bases. An inaccessible or ambiguous
// member (a list, or a class derived from several properties) is a
// substitution failure, hence `false`.
template <typename T, typename = void>
struct is_exact_property : std::false_type {};
template <typename T>
struct is_exact_property<T, std::void_t<typename T::__detail_property_self_t>>
    : std::is_same<T, typename T::__detail_property_self_t> {};

template <typename T, typename = void>
struct is_exact_property_key : std::false_type {};
template <typename T>
struct is_exact_property_key<T, std::void_t<typename T::__detail_key_self_t>>
    : std::is_same<T, typename T::__detail_key_self_t> {};

template <typename T, bool = is_exact_property_key<T>::value>
struct is_compile_time_key : std::false_type {};
template <typename T>
struct is_compile_time_key<T, true>
    : std::bool_constant<T::__detail_is_compile_time> {};

} // namespace detail
} // namespace property

// Classification traits.

template <typename T>
struct is_property
    : std::bool_constant<
          property::detail::is_exact_property<std::remove_cv_t<T>>::value> {};
template <typename T>
inline constexpr bool is_property_v = is_property<T>::value;

template <typename T>
struct is_property_key
    : std::bool_constant<
          property::detail::is_exact_property_key<std::remove_cv_t<T>>::value> {
};
template <typename T>
inline constexpr bool is_property_key_v = is_property_key<T>::value;

template <typename T>
struct is_property_key_compile_time
    : std::bool_constant<
          property::detail::is_compile_time_key<std::remove_cv_t<T>>::value> {};
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

namespace property::detail {
template <typename T, typename Class, typename = void>
struct is_property_for_impl : std::false_type {};
template <typename T, typename Class>
struct is_property_for_impl<T, Class, std::enable_if_t<is_property_v<T>>>
    : std::bool_constant<
          is_property_key_for_v<typename T::__detail_key_t, Class>> {};
} // namespace property::detail

// A property is for `Class` iff its key is for `Class`.
template <typename T, typename Class>
struct is_property_for : property::detail::is_property_for_impl<T, Class> {};
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

namespace property::detail {
// A single property or a property list that is for `Class`; the form used by
// the constraints of every property-accepting entry point.
template <typename T, typename Class>
inline constexpr bool is_property_or_list_for_v =
    is_property_for_v<T, Class> || is_property_list_for_v<T, Class>;

template <typename T, typename... Ts>
inline constexpr int count_same = (0 + ... + int(std::is_same_v<T, Ts>));

// True if no two of `Ps` share a key. Keys are only inspected once every
// element is known to be a property; non-properties are diagnosed separately.
template <bool AllProperties, typename... Ps>
struct unique_keys : std::true_type {};
template <typename... Ps>
struct unique_keys<true, Ps...>
    : std::bool_constant<((count_same<typename Ps::__detail_key_t,
                                      typename Ps::__detail_key_t...> == 1) &&
                          ...)> {};
template <typename... Ps>
inline constexpr bool has_unique_keys =
    unique_keys<(is_property_v<Ps> && ...), Ps...>::value;

// True if `Ps` are properties with pairwise distinct keys.
template <typename... Ps>
inline constexpr bool valid_properties =
    (is_property_v<Ps> && ...) && has_unique_keys<Ps...>;

// An invalid list gets empty storage so that only the mandate diagnostics are
// reported.
template <typename... All>
using storage_for =
    typename std::conditional_t<valid_properties<All...>,
                                build_storage<property_storage<>, All...>,
                                std::enable_if<true, property_storage<>>>::type;

// Constructor constraint: every argument is a property and, for a valid list,
// the arguments are exactly the encoded properties, in any order.
template <typename List, typename... Args> struct ctor_args_match;
template <typename... Encoded, typename... Args>
struct ctor_args_match<properties<Encoded...>, Args...>
    : std::bool_constant<(is_property_v<Args> && ...) &&
                         (!valid_properties<Encoded...> ||
                          (sizeof...(Encoded) == sizeof...(Args) &&
                           ((count_same<Encoded, Args...> == 1) && ...)))> {};
} // namespace property::detail

// The properties container.
template <typename... EncodedProperties>
class __SYCL_EBO properties
    : private property::detail::storage_for<EncodedProperties...> {
  using storage_t = property::detail::storage_for<EncodedProperties...>;

  static_assert((is_property_v<EncodedProperties> && ...),
                "Template arguments of khr::properties must be properties.");

  // Mandate: no two properties may share the same key.
  static_assert(property::detail::has_unique_keys<EncodedProperties...>,
                "Duplicate properties in property list.");

public:
  template <typename... Properties,
            std::enable_if_t<property::detail::ctor_args_match<
                                 properties, Properties...>::value,
                             int> = 0>
  constexpr properties(Properties... props) : storage_t(props...) {}

  template <typename PropertyKey,
            std::enable_if_t<is_property_key_v<PropertyKey>, int> = 0>
  static constexpr bool has_property() {
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

// The concrete named properties defined by this extension are declared in the
// SYCL object headers they apply to (queue.hpp, context.hpp, usm.hpp, ...),
// which include this engine header. This keeps the engine free of the
// dependencies those properties may carry.

} // namespace khr
} // namespace _V1
} // namespace sycl

#endif // __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
