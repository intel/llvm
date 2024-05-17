//==---------- properties.hpp --- SYCL extended property list --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>       // for IsRuntimePr...
#include <sycl/ext/oneapi/properties/property_utils.hpp> // for sorting...
#include <sycl/ext/oneapi/properties/property_value.hpp> // for property_value
#include <sycl/types.hpp>                                // for is_device_c...

#include <sycl/detail/boost/mp11/bind.hpp> //for mp_bind_back
#include <sycl/detail/boost/mp11/map.hpp>  // for mp_is_map

#include <type_traits> // for enable_if_t

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

namespace detail {
using sycl::detail::type_list;
}

namespace detail {
// create map entry for property value
template <class V> struct make_entry {
  // for runtime property: list containing key/value (is same)
  using type = type_list<V>;
};
template <class... T> struct make_entry<property_value<T...>> {
  // for type compile-time property: property value already is type-list
  using type = property_value<T...>;
};
template <class V> using make_entry_t = typename make_entry<V>::type;
// create property map from properties type
template <class P>
using property_map = mp11::mp_transform<make_entry_t, mp11::mp_front<P>>;
template <class T, class = void>
struct is_empty_or_incomplete : std::true_type {};
template <class T>
struct is_empty_or_incomplete<
    T, std::enable_if_t<(sizeof(T) > 0) && !std::is_empty_v<T>>>
    : std::false_type {};
template <class T>
constexpr bool is_empty_or_incomplete_v = is_empty_or_incomplete<T>::value;
template <class T>
using IsRuntimeProperty = mp11::mp_not<is_empty_or_incomplete<T>>;
} // namespace detail

// base case is only for invalid cases to suppress compiler errors about base
// class storage of properties
template <class L, class = void> class properties {
  static_assert(detail::mp11::mp_all_of<L, std::is_class>(),
                "Unrecognized property in property list.");
  static_assert(detail::mp11::mp_is_set<L>(),
                "Duplicate properties in property list.");
};

template <template <class...> class TL, class... V>
class properties<TL<V...>,
                 std::enable_if_t<(std::is_class_v<V> && ...) &&
                                  detail::mp11::mp_is_set<TL<V...>>::value>>
    : V... {
private:
  using map = detail::property_map<properties>; // map of properties
  static_assert(detail::mp11::mp_is_map<map>(),
                "Duplicate properties in property list.");
  using keys = detail::mp11::mp_map_keys<map>;
  using conflicting = detail::mp11::mp_apply<
      detail::mp11::mp_append,
      detail::mp11::mp_transform<detail::ConflictingProperties, keys>>;
  static_assert(
      !detail::mp11::mp_any_of_q<
          conflicting, detail::mp11::mp_bind_front<detail::mp11::mp_set_contains, keys>>(),
      "Conflicting properties in property list.");

public:
  template <class... T>
  constexpr properties(T... v)
      : T(v)... {} // T might have different ordering than V
  template <class P> static constexpr bool has_property() {
    return detail::mp11::mp_map_contains<map, P>();
  }
  template <class P> static constexpr auto get_property(int = 0) {
    using T = detail::mp11::mp_map_find<map, P>;
    static_assert(!std::is_same_v<T, void>,
                  "Property list does not contain the requested property.");
    return T();
  }
  template <class P,
            class = std::enable_if_t<!detail::is_empty_or_incomplete_v<P>>>
  constexpr P get_property() const {
    static_assert(has_property<P>(),
                  "Property list does not contain the requested property.");
    return *this;
  }
};

#ifdef __cpp_deduction_guides
// Deduction guides
template <class... PropertyValueTs>
properties(PropertyValueTs... props)
    -> properties<
        detail::sort_properties<detail::type_list<PropertyValueTs...>>>;
#endif

using empty_properties_t = properties<detail::type_list<>>;

namespace detail {

// Helper for reconstructing a properties type. This assumes that
// PropertyValueTs is sorted and contains only valid properties.
template <typename... PropertyValueTs>
using properties_t = properties<detail::type_list<PropertyValueTs...>>;

template <typename SyclT, typename PropertiesT>
using all_props_are_keys_of = detail::mp11::mp_all_of_q<
    detail::mp11::mp_first<PropertiesT>,
    detail::mp11::mp_bind_back<ext::oneapi::experimental::is_property_value_of,
                               SyclT>>;

// Helper for merging property lists
template <typename PropA, typename PropB> struct merged_properties {
  using A = property_map<PropA>;
  using B = property_map<PropB>;
  template <class K>
  using val_equal = std::is_same<detail::mp11::mp_map_find<A, K>,
                                 detail::mp11::mp_map_find<B, K>>;
  static_assert(
      detail::mp11::mp_all_of<
          detail::mp11::mp_set_intersection<detail::mp11::mp_map_keys<A>,
                                            detail::mp11::mp_map_keys<B>>,
          val_equal>(),
      "Failed to merge property lists due to conflicting properties.");
  using type = properties<detail::sort_properties<detail::mp11::mp_set_union<
      detail::mp11::mp_front<PropA>, detail::mp11::mp_front<PropB>>>>;
};
template <typename... PropertiesT>
using merged_properties_t = typename merged_properties<PropertiesT...>::type;

template <typename Properties, typename PropertyKey>
struct ValueOrDefault { // TODO: this should be a normal function (no wrapping
                        // in struct) and properties should be passed by value
  template <typename ValT> static constexpr auto get(ValT Default) {
    if constexpr (Properties::template has_property<PropertyKey>())
      return Properties::template get_property<PropertyKey>().value;
    else
      return Default;
  }
};
// Checks if a list of properties contains a property.
template <typename PropT, typename PropertiesT>
using ContainsProperty =
    mp11::mp_map_contains<property_map<type_list<PropertiesT>>, PropT>;

} // namespace detail

// Property list traits
template <typename propertiesT> struct is_property_list : std::false_type {};
template <typename PropertyValueTs>
struct is_property_list<properties<PropertyValueTs>>
    : detail::properties_are_sorted<PropertyValueTs> {};

#if __cplusplus > 201402L
template <typename propertiesT>
inline constexpr bool is_property_list_v = is_property_list<propertiesT>::value;
#endif

} // namespace ext::oneapi::experimental

template <typename PropertiesT>
struct is_device_copyable<ext::oneapi::experimental::properties<PropertiesT>>
    : ext::oneapi::experimental::detail::mp11::mp_all_of<PropertiesT,
                                                         is_device_copyable> {};
} // namespace _V1
} // namespace sycl
