//==---------- properties.hpp --- SYCL extended property list --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/property_helper.hpp>
#include <CL/sycl/types.hpp>
#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_utils.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

#include <tuple>
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace experimental {

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
    : sycl::detail::conditional_t<
          IsRuntimeProperty<T>::value,
          PrependTuple<
              T, typename RuntimePropertyStorage<std::tuple<Ts...>>::type>,
          RuntimePropertyStorage<std::tuple<Ts...>>> {};

// Helper class to extract a subset of elements from a tuple.
// NOTES: This assumes no duplicate properties and that all properties in the
//        struct template argument appear in the tuple passed to Extract.
template <typename PropertiesT> struct ExtractProperties {};
template <typename... PropertiesTs>
struct ExtractProperties<std::tuple<PropertiesTs...>> {
  template <typename... PropertyValueTs>
  using ExtractedPropertiesT = std::tuple<>;

  template <typename... PropertyValueTs>
  static ExtractedPropertiesT<PropertyValueTs...>
  Extract(std::tuple<PropertyValueTs...>) {
    return {};
  }
};
template <typename PropertyT, typename... PropertiesTs>
struct ExtractProperties<std::tuple<PropertyT, PropertiesTs...>> {
  template <typename... PropertyValueTs>
  using NextExtractedPropertiesT =
      typename ExtractProperties<std::tuple<PropertiesTs...>>::
          template ExtractedPropertiesT<PropertyValueTs...>;
  template <typename... PropertyValueTs>
  using ExtractedPropertiesT =
      typename PrependTuple<PropertyT,
                            NextExtractedPropertiesT<PropertyValueTs...>>::type;

  template <typename... PropertyValueTs>
  static ExtractedPropertiesT<PropertyValueTs...>
  Extract(std::tuple<PropertyValueTs...> PropertyValues) {
    PropertyT ThisExtractedProperty = std::get<PropertyT>(PropertyValues);
    NextExtractedPropertiesT<PropertyValueTs...> NextExtractedProperties =
        ExtractProperties<std::tuple<PropertiesTs...>>::template Extract<
            PropertyValueTs...>(PropertyValues);
    return std::tuple_cat(std::tuple<PropertyT>{ThisExtractedProperty},
                          NextExtractedProperties);
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

public:
  template <typename... PropertyValueTs>
  properties(PropertyValueTs... props)
      : Storage(detail::ExtractProperties<StorageT>::Extract(
            std::tuple<PropertyValueTs...>{props...})) {}

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

} // namespace experimental
} // namespace oneapi
} // namespace ext

// If property_list is not trivially copyable, allow properties to propagate
// is_device_copyable
template <typename PropertiesT>
struct is_device_copyable<
    ext::oneapi::experimental::properties<PropertiesT>,
    std::enable_if_t<!std::is_trivially_copyable<
        ext::oneapi::experimental::properties<PropertiesT>>::value>>
    : is_device_copyable<PropertiesT> {};
template <typename PropertiesT>
struct is_device_copyable<
    const ext::oneapi::experimental::properties<PropertiesT>,
    std::enable_if_t<!std::is_trivially_copyable<
        const ext::oneapi::experimental::properties<PropertiesT>>::value>>
    : is_device_copyable<PropertiesT> {};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
