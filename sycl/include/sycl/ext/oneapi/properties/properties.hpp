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

// Filters for all runtime properties with data in a tuple of properties.
// NOTE: We only need storage for runtime properties with data.
template <typename T> struct RuntimePropertyStorage {};
template <typename... Ts> struct RuntimePropertyStorage<std::tuple<Ts...>> {
  using type = std::tuple<>;
};
template <typename T, typename... Ts>
struct RuntimePropertyStorage<std::tuple<T, Ts...>>
    : std::conditional_t<IsRuntimeProperty<T>::value,
                         PrependTuple<T, typename RuntimePropertyStorage<
                                             std::tuple<Ts...>>::type>,
                         RuntimePropertyStorage<std::tuple<Ts...>>> {};

// Count occurrences of a type in a tuple.
template <typename T, typename Tuple> struct CountTypeInTuple;
template <typename T, typename... TupleTs>
struct CountTypeInTuple<T, std::tuple<TupleTs...>>
    : std::integral_constant<
          size_t, (0 + ... + static_cast<size_t>(std::is_same_v<T, TupleTs>))> {
};

// Helper for counting the number of properties that are also in PropertyArgsT.
template <typename PropertyArgsT, typename Props> struct CountContainedProps;
template <typename PropertyArgsT>
struct CountContainedProps<PropertyArgsT, std::tuple<>>
    : std::integral_constant<size_t, 0> {};
template <typename PropertyArgsT, typename PropertyT, typename... PropertyTs>
struct CountContainedProps<PropertyArgsT,
                           std::tuple<PropertyT, PropertyTs...>> {
  static constexpr size_t NumOccurrences =
      CountTypeInTuple<PropertyT, PropertyArgsT>::value;
  static_assert(NumOccurrences <= 1,
                "Duplicate occurrences of property in constructor arguments.");
  static constexpr size_t value =
      CountContainedProps<PropertyArgsT, std::tuple<PropertyTs...>>::value +
      NumOccurrences;
};

// Helper class to extract a subset of elements from a tuple.
// NOTES: This assumes no duplicate properties and that all properties in the
//        struct template argument appear in the tuple passed to Extract.
template <typename PropertyArgsT, typename PropertiesT>
struct ExtractProperties;
template <typename PropertyArgsT>
struct ExtractProperties<PropertyArgsT, std::tuple<>> {
  static constexpr std::tuple<> Extract(const PropertyArgsT &) {
    return std::tuple<>{};
  }
};
template <typename PropertyArgsT, typename PropertyT, typename... PropertiesTs>
struct ExtractProperties<PropertyArgsT,
                         std::tuple<PropertyT, PropertiesTs...>> {
  static constexpr std::tuple<PropertyT, PropertiesTs...>
  Extract(const PropertyArgsT &PropertyValues) {
    // TODO: NumOccurrences and checks should be moved out of the function once
    //       https://github.com/intel/llvm/issues/13677 has been fixed.
    constexpr size_t NumOccurrences =
        CountTypeInTuple<PropertyT, PropertyArgsT>::value;
    static_assert(
        NumOccurrences <= 1,
        "Duplicate occurrences of property in constructor arguments.");
    static_assert(NumOccurrences == 1 ||
                      std::is_default_constructible_v<PropertyT>,
                  "Each property in the property list must either be given an "
                  "argument in the constructor or be default-constructible.");

    auto NextExtractedProperties =
        ExtractProperties<PropertyArgsT, std::tuple<PropertiesTs...>>::Extract(
            PropertyValues);

    if constexpr (NumOccurrences == 1) {
      return std::tuple_cat(
          std::tuple<PropertyT>{std::get<PropertyT>(PropertyValues)},
          NextExtractedProperties);
    } else {
      return std::tuple_cat(std::tuple<PropertyT>{PropertyT{}},
                            NextExtractedProperties);
    }
  }
};

} // namespace detail

template <typename PropertiesT> class properties {
  static_assert(detail::IsTuple<PropertiesT>::value,
                "Properties must be in a tuple.");
  static_assert(detail::AllPropertyValues<PropertiesT>::value,
                "Unrecognized property in property list.");
  static_assert(detail::IsSorted<PropertiesT>::value,
                "Properties in property list are not sorted.");
  static_assert(detail::SortedAllUnique<PropertiesT>::value,
                "Duplicate properties in property list.");
  static_assert(detail::NoConflictingProperties<PropertiesT>::value,
                "Conflicting properties in property list.");

public:
  template <typename... PropertyValueTs,
            std::enable_if_t<detail::AllPropertyValues<
                                 std::tuple<PropertyValueTs...>>::value,
                             int> = 0>
  constexpr properties(PropertyValueTs... props)
      : Storage(detail::ExtractProperties<std::tuple<PropertyValueTs...>,
                                          StorageT>::Extract({props...})) {
    // Default-constructible properties do not need to be in the arguments.
    // For properties with a storage, default-constructibility is checked in
    // ExtractProperties, while those without are so by default. As such, all
    // arguments must be a unique property type and must be in PropertiesT.
    constexpr size_t NumContainedProps =
        detail::CountContainedProps<std::tuple<PropertyValueTs...>,
                                    PropertiesT>::value;
    static_assert(NumContainedProps == sizeof...(PropertyValueTs),
                  "One or more property argument is not a property in the "
                  "property list.");
  }

  template <typename PropertyT>
  static constexpr std::enable_if_t<detail::IsProperty<PropertyT>::value, bool>
  has_property() {
    return detail::ContainsProperty<PropertyT, PropertiesT>::value;
  }

  template <typename PropertyT>
  typename std::enable_if_t<detail::IsRuntimeProperty<PropertyT>::value &&
                                has_property<PropertyT>(),
                            PropertyT>
  get_property() const {
    return std::get<PropertyT>(Storage);
  }

  template <typename PropertyT>
  typename std::enable_if_t<detail::IsRuntimeProperty<PropertyT>::value &&
                                !has_property<PropertyT>(),
                            void>
  get_property() const {
    static_assert(has_property<PropertyT>(),
                  "Property list does not contain the requested property.");
    return;
  }

  template <typename PropertyT>
  static constexpr auto get_property(
      typename std::enable_if_t<detail::IsCompileTimeProperty<PropertyT>::value>
          * = 0) {
    static_assert(has_property<PropertyT>(),
                  "Property list does not contain the requested property.");
    return detail::get_property<PropertyT, has_property<PropertyT>(),
                                PropertiesT>();
  }

private:
  using StorageT = typename detail::RuntimePropertyStorage<PropertiesT>::type;

  StorageT Storage;
};

#ifdef __cpp_deduction_guides
// Deduction guides
template <typename... PropertyValueTs>
properties(PropertyValueTs... props)
    -> properties<typename detail::Sorted<PropertyValueTs...>::type>;
#endif

using empty_properties_t = properties<std::tuple<>>;

// Property list traits
template <typename propertiesT> struct is_property_list : std::false_type {};
template <typename... PropertyValueTs>
struct is_property_list<properties<std::tuple<PropertyValueTs...>>>
    : std::is_same<
          properties<std::tuple<PropertyValueTs...>>,
          properties<typename detail::Sorted<PropertyValueTs...>::type>> {};

#if __cplusplus > 201402L
template <typename propertiesT>
inline constexpr bool is_property_list_v = is_property_list<propertiesT>::value;
#endif

namespace detail {

// Helper for reconstructing a properties type. This assumes that
// PropertyValueTs is sorted and contains only valid properties.
template <typename... PropertyValueTs>
using properties_t = properties<std::tuple<PropertyValueTs...>>;

// Helper for merging two property lists;
template <typename LHSPropertiesT, typename RHSPropertiesT>
struct merged_properties;
template <typename... LHSPropertiesTs, typename... RHSPropertiesTs>
struct merged_properties<properties_t<LHSPropertiesTs...>,
                         properties_t<RHSPropertiesTs...>> {
  using type = properties<typename MergeProperties<
      std::tuple<LHSPropertiesTs...>, std::tuple<RHSPropertiesTs...>>::type>;
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

template <typename PropertiesT>
struct is_device_copyable<ext::oneapi::experimental::properties<PropertiesT>>
    : is_device_copyable<PropertiesT> {};
} // namespace _V1
} // namespace sycl
