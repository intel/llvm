//==-- annotated_ptr_properties.hpp - Specific properties of annotated_ptr
//--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/experimental/common_annotated_properties/properties.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp> // for properties_t
#include <sycl/usm/usm_enums.hpp>

#include <type_traits> // for false_type, con...
#include <utility>     // for declval

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {

template <typename T, typename PropertyListT> class annotated_ptr;

struct usm_kind_key
    : detail::compile_time_property_key<detail::PropKind::UsmKind> {
  template <sycl::usm::alloc Kind>
  using value_t =
      property_value<usm_kind_key,
                     std::integral_constant<sycl::usm::alloc, Kind>>;
};

template <sycl::usm::alloc Kind>
inline constexpr usm_kind_key::value_t<Kind> usm_kind;
inline constexpr usm_kind_key::value_t<sycl::usm::alloc::device>
    usm_kind_device;
inline constexpr usm_kind_key::value_t<sycl::usm::alloc::host> usm_kind_host;
inline constexpr usm_kind_key::value_t<sycl::usm::alloc::shared>
    usm_kind_shared;

template <typename T, sycl::usm::alloc Kind>
struct is_valid_property<T, usm_kind_key::value_t<Kind>>
    : std::bool_constant<std::is_pointer<T>::value> {};

template <typename T, typename PropertyListT>
struct is_property_key_of<usm_kind_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

namespace detail {
template <sycl::usm::alloc Kind>
struct PropertyMetaInfo<usm_kind_key::value_t<Kind>> {
  static constexpr const char *name = "sycl-usm-kind";
  static constexpr sycl::usm::alloc value = Kind;
};

template <typename PropertyListT> struct IsUsmKindDevice : std::false_type {};
template <typename... Props>
struct IsUsmKindDevice<detail::properties_t<Props...>>
    : detail::ContainsProperty<std::remove_const_t<decltype(usm_kind_device)>,
                               std::tuple<Props...>> {};

template <typename PropertyListT> struct IsUsmKindHost : std::false_type {};
template <typename... Props>
struct IsUsmKindHost<detail::properties_t<Props...>>
    : detail::ContainsProperty<std::remove_const_t<decltype(usm_kind_host)>,
                               std::tuple<Props...>> {};

template <typename PropertyListT> struct IsUsmKindShared : std::false_type {};
template <typename... Props>
struct IsUsmKindShared<detail::properties_t<Props...>>
    : detail::ContainsProperty<std::remove_const_t<decltype(usm_kind_shared)>,
                               std::tuple<Props...>> {};

} // namespace detail

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
