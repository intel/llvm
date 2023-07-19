//==-------- alloc_util.hpp - Utilities for annotated usm allocation -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <numeric>
#include <sycl/ext/oneapi/annotated_arg/properties.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {

using alloc = sycl::usm::alloc;
using usm_buffer_location =
    ext::intel::experimental::property::usm::buffer_location;

////
//  Type traits for USM allocation with property support
////

// Check if a property list contains the compile-time property alignment
template <typename... Props>
struct HasAlign
    : detail::ContainsProperty<alignment_key, std::tuple<Props...>> {};

// Get the value of compile-time constant alignment from a property list
// The alignement is 0 by default
template <typename PropertyListT> struct GetAlignFromPropList {};

template <typename... Props>
struct GetAlignFromPropList<detail::properties_t<Props...>> {
  using prop_val_t =
      std::conditional_t<HasAlign<Props...>::value,
                         typename detail::FindCompileTimePropertyValueType<
                             alignment_key, std::tuple<Props...>>::type,
                         decltype(alignment<0>)>;
  static constexpr size_t value =
      detail::PropertyMetaInfo<std::remove_const_t<prop_val_t>>::value;
};

// Check if a property list contains the compile-time property usm_kind
template <typename PropertyListT> struct HasUsmKind : std::false_type {};

template <typename... Props>
struct HasUsmKind<detail::properties_t<Props...>>
    : detail::ContainsProperty<usm_kind_key, std::tuple<Props...>> {};

// Get the value of compile-time constant usm_kind from a property list
// The usm_kind is sycl::usm::alloc::unknown by default
template <typename PropertyListT> struct GetUsmKindFromPropList {};

template <typename... Props>
struct GetUsmKindFromPropList<detail::properties_t<Props...>> {
  using prop_val_t =
      std::conditional_t<HasUsmKind<detail::properties_t<Props...>>::value,
                         typename detail::FindCompileTimePropertyValueType<
                             usm_kind_key, std::tuple<Props...>>::type,
                         decltype(usm_kind<alloc::unknown>)>;
  static constexpr alloc value =
      detail::PropertyMetaInfo<std::remove_const_t<prop_val_t>>::value;
};

// Check if a property list contains the compile-time property buffer_location
template <typename PropertyListT> struct HasBufferLocation : std::false_type {};

template <typename... Props>
struct HasBufferLocation<detail::properties_t<Props...>>
    : detail::ContainsProperty<buffer_location_key, std::tuple<Props...>> {};

// Get the value of compile-time constant buffer_location from a property list
// The buffer location is -1 by default
template <typename PropertyListT> struct GetBufferLocationFromPropList {};

template <typename... Props>
struct GetBufferLocationFromPropList<detail::properties_t<Props...>> {
  using prop_val_t = std::conditional_t<
      HasBufferLocation<detail::properties_t<Props...>>::value,
      typename detail::FindCompileTimePropertyValueType<
          buffer_location_key, std::tuple<Props...>>::type,
      decltype(buffer_location<-1>)>;
  static constexpr int value =
      detail::PropertyMetaInfo<std::remove_const_t<prop_val_t>>::value;
};

// Filter the compile-time properties from a property list
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

// Given the input property list of annotated USM allocation functions, generate
// the property list for the annotated_ptr output of
template <alloc Kind, typename PropertyListT>
struct GetAnnotatedPtrPropertiesWithUsmKind {};

// Partial specialization
template <alloc Kind, typename... Props>
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

// Validate the template arguments for annotated USM allocation functions, which
// consist of
//   allocated data type: T
//   input property list: propertyListA
//   property list for output anntoated_ptr: propertyListB
template <typename T, typename propertyListA, typename propertyListB>
struct CheckTAndPropLists : std::false_type {};

template <typename T, typename... PropsA, typename... PropsB>
struct CheckTAndPropLists<T, detail::properties_t<PropsA...>,
                          detail::properties_t<PropsB...>>
    : std::conditional_t<
          is_property_list<T>::value, std::false_type,
          std::is_same<detail::properties_t<PropsB...>,
                       typename GetCompileTimeProperties<
                           detail::properties_t<PropsA...>>::type>> {};

template <alloc Kind, typename T, typename propertyListA,
          typename propertyListB>
struct CheckTAndPropListsWithUsmKind : std::false_type {};

template <alloc Kind, typename T, typename... PropsA, typename... PropsB>
struct CheckTAndPropListsWithUsmKind<Kind, T, detail::properties_t<PropsA...>,
                                     detail::properties_t<PropsB...>>
    : std::conditional_t<
          is_property_list<T>::value, std::false_type,
          std::is_same<detail::properties_t<PropsB...>,
                       typename GetAnnotatedPtrPropertiesWithUsmKind<
                           Kind, detail::properties_t<PropsA...>>::type>> {};

////
//  Utility functions for USM allocation with property support
////

// Transform a compile-time property list to a USM property_list (working at
// runtime). Right now only the `buffer_location<N>` has its corresponding USM
// runtime property and is transformable
template <typename PropertyListT>
inline property_list get_usm_property_list(const PropertyListT &propList) {
  if constexpr (HasBufferLocation<PropertyListT>::value) {
    return property_list{usm_buffer_location(
        GetBufferLocationFromPropList<PropertyListT>::value)};
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

// When allocating USM device memory, the device must have
// aspect::usm_device_allocations
inline void check_device_aspect(const device &syclDevice) {
  if (!syclDevice.has(aspect::usm_device_allocations)) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Can not allocate USM device memroy if the device does not have "
        "aspect::usm_device_allocations.");
  }
}

// When allocating USM shared memory, the device must have
// aspect::usm_shared_allocations
inline void check_shared_aspect(const device &syclDevice) {
  if (!syclDevice.has(aspect::usm_shared_allocations)) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Can not allocate USM device memroy if the device does not have "
        "aspect::usm_shared_allocations.");
  }
}

// When allocating USM host memory, at least one device in the allocation
// context must have aspect::usm_host_allocations
inline void check_host_aspect(const context &syclContext) {
  std::vector<device> devs = syclContext.get_devices();
  if (devs.size() > 0 &&
      std::none_of(devs.begin(), devs.end(), [](const device &d) {
        return d.has(aspect::usm_host_allocations);
      })) {

    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Can not allocate USM device memroy if none of the devices in the "
        "allocation context have "
        "aspect::usm_host_allocations.");
  }
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl