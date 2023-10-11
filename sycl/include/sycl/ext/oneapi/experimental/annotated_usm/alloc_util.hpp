//==-------- alloc_util.hpp - Utilities for annotated usm allocation -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <numeric>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {

namespace detail {

////
//  Type traits for USM allocation with property support
////

// Merge a property list with the usm_kind property
template <sycl::usm::alloc Kind, typename PropertyListT>
using MergeUsmKind =
    detail::merged_properties_t<PropertyListT,
                                decltype(properties{usm_kind<Kind>})>;

// Check if a property list contains the a certain property
template <typename PropKey, typename PropertyListT> struct HasProperty {};

template <typename PropKey, typename... Props>
struct HasProperty<PropKey, detail::properties_t<Props...>>
    : detail::ContainsProperty<PropKey, std::tuple<Props...>> {};

template <typename PropertyListT>
using HasAlign = HasProperty<alignment_key, PropertyListT>;
template <typename PropertyListT>
using HasUsmKind = HasProperty<usm_kind_key, PropertyListT>;
template <typename PropertyListT>
using HasBufferLocation = HasProperty<buffer_location_key, PropertyListT>;

// Get the value of a property from a property list
template <typename PropKey, typename ConstType, typename DefaultPropVal,
          typename PropertyListT>
struct GetPropertyValueFromPropList {};

template <typename PropKey, typename ConstType, typename DefaultPropVal,
          typename... Props>
struct GetPropertyValueFromPropList<PropKey, ConstType, DefaultPropVal,
                                    detail::properties_t<Props...>> {
  using prop_val_t = std::conditional_t<
      detail::ContainsProperty<PropKey, std::tuple<Props...>>::value,
      typename detail::FindCompileTimePropertyValueType<
          PropKey, std::tuple<Props...>>::type,
      DefaultPropVal>;
  static constexpr ConstType value =
      detail::PropertyMetaInfo<std::remove_const_t<prop_val_t>>::value;
};

// Get the value of alignment from a property list
// If alignment is not present in the property list, set to default value 0
template <typename PropertyListT>
using GetAlignFromPropList =
    GetPropertyValueFromPropList<alignment_key, size_t, decltype(alignment<0>),
                                 PropertyListT>;
// Get the value of usm_kind from a property list
// The usm_kind is sycl::usm::alloc::unknown by default
template <typename PropertyListT>
using GetUsmKindFromPropList =
    GetPropertyValueFromPropList<usm_kind_key, sycl::usm::alloc,
                                 decltype(usm_kind<sycl::usm::alloc::unknown>),
                                 PropertyListT>;
// Get the value of buffer_location from a property list
// The buffer location is -1 by default
template <typename PropertyListT>
using GetBufferLocationFromPropList = GetPropertyValueFromPropList<
    buffer_location_key, int,
    decltype(sycl::ext::intel::experimental::buffer_location<-1>),
    PropertyListT>;

// Check if a runtime property is valid
template <typename Prop> struct IsRuntimePropertyValid : std::false_type {};

// Validate the property list of annotated USM allocations by checking each
// property is
// 1. a valid compile-time property for annotated_ptr, or
// 2. a valid runtime property for annotated USM allocations
template <typename T, typename propertyList>
struct ValidAllocPropertyList : std::false_type {};
template <typename T>
struct ValidAllocPropertyList<T, detail::empty_properties_t> : std::true_type {
};
template <typename T, typename Prop, typename... Props>
struct ValidAllocPropertyList<T, detail::properties_t<Prop, Props...>>
    : std::integral_constant<
          bool, (is_valid_property<T *, Prop>::value ||
                 IsRuntimePropertyValid<Prop>::value) &&
                    ValidAllocPropertyList<
                        T, detail::properties_t<Props...>>::value> {
  // check if a compile-time property is valid for annotated_ptr
  static_assert(!detail::IsCompileTimePropertyValue<Prop>::value ||
                    is_valid_property<T *, Prop>::value,
                "Found invalid compile-time property in the property list.");
  // check if a runtime property is valid for malloc
  static_assert(!detail::IsRuntimeProperty<Prop>::value ||
                    IsRuntimePropertyValid<Prop>::value,
                "Found invalid runtime property in the property list.");
};

// The utility to filter a given property list to get the properties for
// annotated_ptr
template <typename PropertyListT> struct GetCompileTimeProperties {};

template <> struct GetCompileTimeProperties<detail::empty_properties_t> {
  using type = detail::empty_properties_t;
};

template <typename Prop>
struct GetCompileTimeProperties<detail::properties_t<Prop>> {
  using type =
      std::conditional_t<detail::IsCompileTimePropertyValue<Prop>::value,
                         detail::properties_t<Prop>,
                         detail::empty_properties_t>;
};

template <typename Prop, typename... Props>
struct GetCompileTimeProperties<detail::properties_t<Prop, Props...>> {
  using filtered_this_property_t =
      std::conditional_t<detail::IsCompileTimePropertyValue<Prop>::value,
                         detail::properties_t<Prop>,
                         detail::empty_properties_t>;
  using filtered_other_properties_t =
      typename GetCompileTimeProperties<detail::properties_t<Props...>>::type;
  using type = detail::merged_properties_t<filtered_this_property_t,
                                           filtered_other_properties_t>;
};

// Given the input property list `PropertyListT` and the usm kind `Kind` of
// annotated USM allocation functions, generate the property list for the
// returned annotated_ptr by following the rules:
// 1. all the compile-time properties in PropertyListT should appear in the
// output properties
// 2. if PropertyListT contains usm_kind, the value should be the same as
// `Kind`, usm_kind property should appear in the ouput properties
// 3. if PropertyListT does not contain usm_kind, a usm_kind property with value
// `Kind` is inserted in the ouput properties
template <sycl::usm::alloc Kind, typename PropertyListT>
struct GetAnnotatedPtrPropertiesWithUsmKind {};
template <sycl::usm::alloc Kind, typename... Props>
struct GetAnnotatedPtrPropertiesWithUsmKind<Kind,
                                            detail::properties_t<Props...>> {
  using input_properties_t = detail::properties_t<Props...>;
  using filtered_input_properties_t =
      typename GetCompileTimeProperties<input_properties_t>::type;

  static_assert(!HasUsmKind<input_properties_t>::value ||
                    GetUsmKindFromPropList<input_properties_t>::value == Kind,
                "Input property list contains conflicting USM kind.");

  using type =
      detail::merged_properties_t<filtered_input_properties_t,
                                  decltype(properties{usm_kind<Kind>})>;
};

// Check if the 3 template parameters of annotated USM allocation functions,
// i.e.
//   T: allocated data type
//   propertyListA: input property list
//   propertyListB: property list for output anntoated_ptr
// meet the following conditions:
//  1. T is not a property list
//  2. propertyListB == all the compile-time properties in propertyListA
template <typename T, typename propertyListA, typename propertyListB>
struct CheckTAndPropLists : std::false_type {};

template <typename T, typename... PropsA, typename... PropsB>
struct CheckTAndPropLists<T, detail::properties_t<PropsA...>,
                          detail::properties_t<PropsB...>>
    : std::integral_constant<
          bool,
          !is_property_list<T>::value &&
              std::is_same_v<detail::properties_t<PropsB...>,
                             typename GetCompileTimeProperties<
                                 detail::properties_t<PropsA...>>::type>> {};

// Check if the 3 template parameters of annotated USM allocation functions of a
// certain usm kind:
//   T: allocated data type
//   propertyListA: input property list
//   propertyListB: property list for output anntoated_ptr
// meet the following conditions:
//  1. T is not a property list
//  2. if propertyListA contains usm_kind, the usm_kind must match with the
//  specified usm kind.
//     propertyListB is all the compile-time properties in propertyListA
//  3. if propertyListA does not contain usm_kind, propertyListB is all the
//  compile-time
//     properties in propertyListA with the usm_kind property inserted
template <sycl::usm::alloc Kind, typename T, typename propertyListA,
          typename propertyListB>
struct CheckTAndPropListsWithUsmKind : std::false_type {};

template <sycl::usm::alloc Kind, typename T, typename... PropsA,
          typename... PropsB>
struct CheckTAndPropListsWithUsmKind<Kind, T, detail::properties_t<PropsA...>,
                                     detail::properties_t<PropsB...>>
    : std::integral_constant<
          bool, !is_property_list<T>::value &&
                    std::is_same_v<
                        detail::properties_t<PropsB...>,
                        typename GetAnnotatedPtrPropertiesWithUsmKind<
                            Kind, detail::properties_t<PropsA...>>::type>> {};

} // namespace detail

////
//  Utility functions for USM allocation with property support
////

// Transform a compile-time property list to a USM property_list (working at
// runtime). Right now only the `buffer_location<N>` has its corresponding USM
// runtime property and is transformable
template <typename PropertyListT> inline property_list get_usm_property_list() {
  if constexpr (detail::HasBufferLocation<PropertyListT>::value) {
    return property_list{
        sycl::ext::intel::experimental::property::usm::buffer_location(
            detail::GetBufferLocationFromPropList<PropertyListT>::value)};
  }
  return {};
}

// Combine two alignment requirements for a pointer in the following way:
// 1. if either of the alignments is 0, return the other alignment
// 2. otherwise return the least common multiple of the two alignments
inline size_t combine_align(size_t alignA, size_t alignB) {
  return alignA == 0 ? alignB
                     : (alignB == 0 ? alignA : std::lcm(alignA, alignB));
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl