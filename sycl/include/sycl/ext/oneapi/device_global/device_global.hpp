//==----------- device_global.hpp - SYCL device_global extension -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <type_traits>

#include <sycl/detail/stl_type_traits.hpp>
#include <sycl/exception.hpp>
#include <sycl/ext/oneapi/device_global/properties.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_HOST_NOT_SUPPORTED(Op)
#else
#define __SYCL_HOST_NOT_SUPPORTED(Op)                                          \
  throw sycl::exception(                                                       \
      sycl::make_error_code(sycl::errc::feature_not_supported),                \
      Op " is not supported on host device.");
#endif

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {

namespace detail {
// Type-trait for checking if a type defines `operator->`.
template <typename T, typename = void>
struct HasArrowOperator : std::false_type {};
template <typename T>
struct HasArrowOperator<
    T, sycl::detail::void_t<decltype(std::declval<T>().operator->())>>
    : std::true_type {};

// Base class for device_global.
template <typename T, typename PropertyListT, typename = void>
class device_global_base {
protected:
  T *usmptr;
  T *get_ptr() noexcept { return usmptr; }
  const T *get_ptr() const noexcept { return usmptr; }
};

// Specialization of device_global base class for when device_image_scope is in
// the property list.
template <typename T, typename... Props>
class device_global_base<
    T, properties_t<Props...>,
    sycl::detail::enable_if_t<properties_t<Props...>::template has_property<
        device_image_scope_key>()>> {
protected:
  T val{};
  T *get_ptr() noexcept { return &val; }
  const T *get_ptr() const noexcept { return &val; }
};
} // namespace detail

template <typename T, typename PropertyListT = detail::empty_properties_t>
class
#ifdef __SYCL_DEVICE_ONLY__
    // FIXME: Temporary work-around. Remove when fixed.
    [[__sycl_detail__::global_variable_allowed, __sycl_detail__::device_global]]
#endif
    device_global {
  // This should always fail when instantiating the unspecialized version.
  static_assert(is_property_list<PropertyListT>::value,
                "Property list is invalid.");
};

template <typename T, typename... Props>
class
#ifdef __SYCL_DEVICE_ONLY__
    [[__sycl_detail__::global_variable_allowed, __sycl_detail__::device_global,
      __sycl_detail__::add_ir_attributes_global_variable(
          "sycl-device-global-size", detail::PropertyMetaInfo<Props>::name...,
          sizeof(T), detail::PropertyMetaInfo<Props>::value...)]]
#endif
    device_global<T, detail::properties_t<Props...>>
    : public detail::device_global_base<T, detail::properties_t<Props...>> {

  using property_list_t = detail::properties_t<Props...>;

public:
  using element_type = std::remove_extent_t<T>;

  static_assert(std::is_trivially_default_constructible<T>::value,
                "Type T must be trivially default constructable (until C++20 "
                "consteval is supported and enabled.)");

  static_assert(std::is_trivially_destructible<T>::value,
                "Type T must be trivially destructible.");

  static_assert(is_property_list<property_list_t>::value,
                "Property list is invalid.");

  // TODO: Remove when support has been added for device_global without the
  // device_image_scope property.
  static_assert(
      property_list_t::template has_property<device_image_scope_key>(),
      "device_global without the device_image_scope property is currently "
      "unavailable.");

  device_global() = default;

  device_global(const device_global &) = delete;
  device_global(const device_global &&) = delete;
  device_global &operator=(const device_global &) = delete;
  device_global &operator=(const device_global &&) = delete;

  template <access::decorated IsDecorated>
  multi_ptr<T, access::address_space::global_space, IsDecorated>
  get_multi_ptr() noexcept {
    __SYCL_HOST_NOT_SUPPORTED("get_multi_ptr()")
    return address_space_cast<access::address_space::global_space, IsDecorated>(
        this->get_ptr());
  }

  template <access::decorated IsDecorated>
  multi_ptr<const T, access::address_space::global_space, IsDecorated>
  get_multi_ptr() const noexcept {
    __SYCL_HOST_NOT_SUPPORTED("get_multi_ptr()")
    return address_space_cast<access::address_space::global_space, IsDecorated,
                              const T>(this->get_ptr());
  }

  T &get() noexcept {
    __SYCL_HOST_NOT_SUPPORTED("get()")
    return *this->get_ptr();
  }

  const T &get() const noexcept {
    __SYCL_HOST_NOT_SUPPORTED("get()")
    return *this->get_ptr();
  }

  operator T &() noexcept {
    __SYCL_HOST_NOT_SUPPORTED("Implicit conversion of device_global to T")
    return get();
  }

  operator const T &() const noexcept {
    __SYCL_HOST_NOT_SUPPORTED("Implicit conversion of device_global to T")
    return get();
  }

  device_global &operator=(const T &newValue) noexcept {
    __SYCL_HOST_NOT_SUPPORTED("Assignment operator")
    *this->get_ptr() = newValue;
    return *this;
  }

  template <class RelayT = T>
  std::remove_reference_t<
      decltype(std::declval<RelayT>()[std::declval<std::ptrdiff_t>()])>
      &operator[](std::ptrdiff_t idx) noexcept {
    __SYCL_HOST_NOT_SUPPORTED("Subscript operator")
    return (*this->get_ptr())[idx];
  }

  template <class RelayT = T>
  const std::remove_reference_t<
      decltype(std::declval<RelayT>()[std::declval<std::ptrdiff_t>()])>
      &operator[](std::ptrdiff_t idx) const noexcept {
    __SYCL_HOST_NOT_SUPPORTED("Subscript operator")
    return (*this->get_ptr())[idx];
  }

  template <class RelayT = T>
  std::enable_if_t<detail::HasArrowOperator<RelayT>::value ||
                       std::is_pointer<RelayT>::value,
                   RelayT>
      &operator->() noexcept {
    __SYCL_HOST_NOT_SUPPORTED("operator-> on a device_global")
    return *this->get_ptr();
  }

  template <class RelayT = T>
  std::enable_if_t<detail::HasArrowOperator<RelayT>::value ||
                       std::is_pointer<RelayT>::value,
                   const RelayT>
      &operator->() const noexcept {
    __SYCL_HOST_NOT_SUPPORTED("operator-> on a device_global")
    return *this->get_ptr();
  }

  template <typename propertyT> static constexpr bool has_property() {
    return property_list_t::template has_property<propertyT>();
  }

  template <typename propertyT> static constexpr auto get_property() {
    return property_list_t::template get_property<propertyT>();
  }
};

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

#undef __SYCL_HOST_NOT_SUPPORTED
