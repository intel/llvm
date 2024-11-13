//==---------- properties.hpp --- SYCL extended property list --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/is_device_copyable.hpp>
#include <sycl/ext/oneapi/properties/property.hpp>       // for IsRuntimePr...
#include <sycl/ext/oneapi/properties/property_utils.hpp> // for Sorted, Mer...
#include <sycl/ext/oneapi/properties/property_value.hpp> // for property_value

#include <tuple>       // for tuple, tupl...
#include <type_traits> // for enable_if_t
#include <variant>     // for tuple

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

namespace detail {

// Checks if a tuple of properties contains a property.
template <typename PropT, typename PropertiesT>
struct ContainsProperty : std::false_type {};
template <typename PropT, typename T, typename... Ts>
struct ContainsProperty<PropT, std::tuple<T, Ts...>>
    : ContainsProperty<PropT, std::tuple<Ts...>> {};
template <typename PropT, typename... Rest>
struct ContainsProperty<PropT, std::tuple<PropT, Rest...>> : std::true_type {};
template <typename PropT, typename... PropValuesTs, typename... Rest>
struct ContainsProperty<
    PropT, std::tuple<property_value<PropT, PropValuesTs...>, Rest...>>
    : std::true_type {};

// Finds the full property_value type of a property in a tuple of properties.
// type is void if the type was not found in the tuple of properties.
template <typename CTPropertyT, typename PropertiesT = void>
struct FindCompileTimePropertyValueType {
  using type = void;
};
template <typename CTPropertyT, typename OtherProperty, typename... Rest>
struct FindCompileTimePropertyValueType<CTPropertyT,
                                        std::tuple<OtherProperty, Rest...>> {
  using type =
      typename FindCompileTimePropertyValueType<CTPropertyT,
                                                std::tuple<Rest...>>::type;
};
template <typename CTPropertyT, typename... CTPropertyValueTs, typename... Rest>
struct FindCompileTimePropertyValueType<
    CTPropertyT,
    std::tuple<property_value<CTPropertyT, CTPropertyValueTs...>, Rest...>> {
  using type = property_value<CTPropertyT, CTPropertyValueTs...>;
};

template <typename CTPropertyT, bool HasProperty, typename PropertiesT = void>
static constexpr std::enable_if_t<
    HasProperty,
    typename FindCompileTimePropertyValueType<CTPropertyT, PropertiesT>::type>
get_property() {
  return {};
}

template <typename CTPropertyT, bool HasProperty, typename PropertiesT = void>
static constexpr std::enable_if_t<!HasProperty, void> get_property() {
  return;
}

// Get the value of a property from a property list
template <typename PropKey, typename ConstType, typename DefaultPropVal,
          typename PropertiesT>
struct GetPropertyValueFromPropList {};

template <typename PropKey, typename ConstType, typename DefaultPropVal,
          typename... PropertiesT>
struct GetPropertyValueFromPropList<PropKey, ConstType, DefaultPropVal,
                                    std::tuple<PropertiesT...>> {
  using prop_val_t = std::conditional_t<
      ContainsProperty<PropKey, std::tuple<PropertiesT...>>::value,
      typename FindCompileTimePropertyValueType<
          PropKey, std::tuple<PropertiesT...>>::type,
      DefaultPropVal>;
  static constexpr ConstType value =
      PropertyMetaInfo<std::remove_const_t<prop_val_t>>::value;
};

template <typename... property_tys>
inline constexpr bool properties_are_unique = []() constexpr {
  if constexpr (sizeof...(property_tys) == 0) {
    return true;
  } else {
    const std::array kinds = {PropertyID<property_tys>::value...};
    auto N = kinds.size();
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = i + 1; j < N; ++j)
        if (kinds[i] == kinds[j])
          return false;

    return true;
  }
}();

template <typename... property_tys>
inline constexpr bool properties_are_sorted = []() constexpr {
  if constexpr (sizeof...(property_tys) == 0) {
    return true;
  } else {
    const std::array kinds = {PropertyID<property_tys>::value...};
    // std::is_sorted isn't constexpr until C++20.
    for (std::size_t idx = 1; idx < kinds.size(); ++idx)
      if (kinds[idx - 1] >= kinds[idx])
        return false;
    return true;
  }
}();

template <typename... property_tys>
constexpr bool properties_are_valid_for_ctad = []() constexpr {
  // Need `if constexpr` to avoid hard error in "unique" check when querying
  // property kind if `property_tys` isn't a property.
  if constexpr (!((is_property_value_v<property_tys> && ...))) {
    return false;
  } else if constexpr (!detail::properties_are_unique<property_tys...>) {
    return false;
  } else {
    return true;
  }
}();

template <typename... property_tys> struct properties_type_list;
template <typename... property_tys> struct invalid_properties_type_list {};
} // namespace detail

template <typename properties_type_list_ty> class __SYCL_EBO properties;

// Empty property list.
template <> class __SYCL_EBO properties<detail::properties_type_list<>> {
  template <typename T>
  static constexpr bool empty_properties_list_contains = false;

public:
  template <typename property_key_t> static constexpr bool has_property() {
    return false;
  }

  // Never exists for empty property list, provide this for a better error
  // message:
  template <typename T>
  static std::enable_if_t<empty_properties_list_contains<T>> get_property() {}
};

// Base implementation to provide nice user error in case of mis-use. Without it
// an error "base class '<property>' specified more than once as a direct base
// class" is reported prior to static_assert's error.
template <typename... property_tys>
class __SYCL_EBO
    properties<detail::invalid_properties_type_list<property_tys...>> {
public:
  properties(property_tys...) {
    if constexpr (!((is_property_value_v<property_tys> && ...))) {
      static_assert(((is_property_value_v<property_tys> && ...)),
                    "Non-property argument!");
    } else {
      // This is a separate specialization to report an error, we can afford
      // doing extra work to provide nice error message without sacrificing
      // compile time on non-exceptional path. Let's find *a* pair of properties
      // that failed the check. Note that there might be multiple duplicate
      // names, we're only reporting one instance. Once user addresses that, the
      // next pair will be reported.
      static constexpr auto conflict = []() constexpr {
        const std::array kinds = {detail::PropertyID<property_tys>::value...};
        auto N = kinds.size();
        for (int i = 0; i < N; ++i)
          for (int j = i + 1; j < N; ++j)
            if (kinds[i] == kinds[j])
              return std::pair{i, j};
      }();
      using first_type = detail::nth_type_t<conflict.first, property_tys...>;
      using second_type = detail::nth_type_t<conflict.second, property_tys...>;
      if constexpr (std::is_same_v<typename first_type::key_t,
                                   typename second_type::key_t>) {
        static_assert(!std::is_same_v<typename first_type::key_t,
                                      typename second_type::key_t>,
                      "Duplicate properties in property list.");
      } else {
        static_assert(
            detail::PropertyToKind<first_type>::Kind !=
                detail::PropertyToKind<second_type>::Kind,
            "Property Kind collision between different property keys!");
      }
    }
  }
};

// NOTE: Meta-function to implement CTAD rules isn't allowed to return
// `properties<something>` and it's impossible to return a pack as well. As
// such, we're forced to have an extra level of `detail::properties_type_list`
// for the purpose of providing CTAD rules.
template <typename... property_tys>
class __SYCL_EBO properties<detail::properties_type_list<property_tys...>>
    : private property_tys... {
  static_assert(detail::properties_are_sorted<property_tys...>,
                "Properties must be sorted!");
  static_assert(
      detail::NoConflictingProperties<std::tuple<property_tys...>>::value,
      "Conflicting properties in property list.");
  using property_tys::get_property_impl...;

  template <typename> friend class __SYCL_EBO properties;

  template <typename prop_t> static constexpr bool is_valid_ctor_arg() {
    return ((std::is_same_v<prop_t, property_tys> || ...));
  }

  template <typename prop_t, typename... unsorted_property_tys>
  static constexpr bool can_be_constructed_from() {
    return std::is_default_constructible_v<prop_t> ||
           ((false || ... || std::is_same_v<prop_t, unsorted_property_tys>));
  }

  // It's possible it shouldn't be that complicated, but clang doesn't accept
  // simpler version: https://godbolt.org/z/oPff4h738, reported upstream at
  // https://github.com/llvm/llvm-project/issues/115547. Note that if the
  // `decltype(...)` is "inlined" then it has no issues with it, but that's too
  // verbose.
  struct helper : property_tys... {
    using property_tys::get_property_impl...;
  };
  template <typename property_key_t>
  using prop_t = decltype(std::declval<helper>().get_property_impl(
      detail::property_key_tag<property_key_t>{}));

public:
  template <
      typename... unsorted_property_tys,
      typename = std::enable_if_t<
          ((is_valid_ctor_arg<unsorted_property_tys>() && ...))>,
      typename = std::enable_if_t<
          ((can_be_constructed_from<property_tys, unsorted_property_tys...>() &&
            ...))>,
      typename = std::enable_if_t<
          detail::properties_are_unique<unsorted_property_tys...>>>
  constexpr properties(unsorted_property_tys... props)
      : unsorted_property_tys(props)... {}

  template <typename property_key_t> static constexpr bool has_property() {
    return std::is_base_of_v<detail::property_key_tag<property_key_t>,
                             properties>;
  }

  // Compile-time property.
  template <typename property_key_t>
  static constexpr auto
  get_property() -> std::enable_if_t<std::is_empty_v<prop_t<property_key_t>>,
                                     prop_t<property_key_t>> {
    return prop_t<property_key_t>{};
  }

  // Runtime property.
  // Extra operand to make MSVC happy as it complains otherwise:
  // https://godbolt.org/z/WGqdqrejj
  template <typename property_key_t>
  constexpr auto get_property(int = 0) const
      -> std::enable_if_t<!std::is_empty_v<prop_t<property_key_t>>,
                          prop_t<property_key_t>> {
    return get_property_impl(detail::property_key_tag<property_key_t>{});
  }
};

// Deduction guides
template <typename... PropertyValueTs,
          typename = std::enable_if_t<
              detail::properties_are_valid_for_ctad<PropertyValueTs...>>>
properties(PropertyValueTs... props)
    -> properties<typename detail::Sorted<PropertyValueTs...>::type>;

template <typename... PropertyValueTs,
          typename = std::enable_if_t<
              !detail::properties_are_valid_for_ctad<PropertyValueTs...>>>
properties(PropertyValueTs... props)
    -> properties<detail::invalid_properties_type_list<PropertyValueTs...>>;

using empty_properties_t = decltype(properties{});

namespace detail {

// Helper for reconstructing a properties type. This assumes that
// PropertyValueTs is sorted and contains only valid properties.
template <typename... PropertyValueTs>
using properties_t =
    properties<detail::properties_type_list<PropertyValueTs...>>;

// Helper for merging two property lists;
template <typename LHSPropertiesT, typename RHSPropertiesT>
struct merged_properties;
template <typename... LHSPropertiesTs, typename... RHSPropertiesTs>
struct merged_properties<properties_t<LHSPropertiesTs...>,
                         properties_t<RHSPropertiesTs...>> {
  using type = properties<
      typename MergeProperties<properties_type_list<LHSPropertiesTs...>,
                               properties_type_list<RHSPropertiesTs...>>::type>;
};
template <typename LHSPropertiesT, typename RHSPropertiesT>
using merged_properties_t =
    typename merged_properties<LHSPropertiesT, RHSPropertiesT>::type;

template <typename Properties, typename PropertyKey, typename Cond = void>
struct ValueOrDefault {
  template <typename ValT> static constexpr ValT get(ValT Default) {
    return Default;
  }
};

template <typename Properties, typename PropertyKey>
struct ValueOrDefault<
    Properties, PropertyKey,
    std::enable_if_t<is_property_list_v<Properties> &&
                     Properties::template has_property<PropertyKey>()>> {
  template <typename ValT> static constexpr ValT get(ValT) {
    return Properties::template get_property<PropertyKey>().value;
  }
};

// helper: check_all_props_are_keys_of
template <typename SyclT> constexpr bool check_all_props_are_keys_of() {
  return true;
}

template <typename SyclT, typename FirstProp, typename... RestProps>
constexpr bool check_all_props_are_keys_of() {
  return ext::oneapi::experimental::is_property_key_of<FirstProp,
                                                       SyclT>::value &&
         check_all_props_are_keys_of<SyclT, RestProps...>();
}

// all_props_are_keys_of
template <typename SyclT, typename PropertiesT>
struct all_props_are_keys_of : std::false_type {};

template <typename SyclT>
struct all_props_are_keys_of<SyclT,
                             ext::oneapi::experimental::empty_properties_t>
    : std::true_type {};

template <typename SyclT, typename PropT>
struct all_props_are_keys_of<
    SyclT, ext::oneapi::experimental::detail::properties_t<PropT>>
    : std::bool_constant<
          ext::oneapi::experimental::is_property_key_of<PropT, SyclT>::value> {
};

template <typename SyclT, typename... Props>
struct all_props_are_keys_of<
    SyclT, ext::oneapi::experimental::detail::properties_t<Props...>>
    : std::bool_constant<check_all_props_are_keys_of<SyclT, Props...>()> {};

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
