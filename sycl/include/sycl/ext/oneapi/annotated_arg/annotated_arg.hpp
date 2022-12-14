//==----------- annotated_arg.hpp - SYCL annotated_arg extension -----------==//
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
#include <sycl/ext/oneapi/annotated_arg/properties.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {

namespace detail {

// Type-trait for checking if a type defines `operator[]`.
template <typename T>
struct HasSubscriptOperator
    : std::bool_constant<
          !std::is_void<decltype(std::declval<T>().operator[](0))>::value>{};

// Deduce a `properties<>` type from given variadic properties
template <typename... Args> struct DeducedProperties {
  using type = decltype(properties{std::declval<Args>()...});
};

// Partial specialization for deducing a `properties<>` type by forwarding the
// given `properties<>` type
template <typename... Args>
struct DeducedProperties<detail::properties_t<Args...>> {
  using type = detail::properties_t<Args...>;
};

} // namespace detail

// Deduction guide
template <typename T, typename... Args>
annotated_arg(T, Args...)
    -> annotated_arg<T, typename detail::DeducedProperties<Args...>::type>;

template <typename T, typename old, typename... ArgT>
annotated_arg(annotated_arg<T, old>, properties<std::tuple<ArgT...>>)
    -> annotated_arg<
        T, detail::merged_properties_t<old, detail::properties_t<ArgT...>>>;

template <typename T, typename PropertyListT = detail::empty_properties_t>
class annotated_arg {
  // This should always fail when instantiating the unspecialized version.
  static_assert(is_property_list<PropertyListT>::value,
                "Property list is invalid.");
};

// Partial specialization for pointer type
template <typename T, typename... Props>
class __SYCL_SPECIAL_CLASS
__SYCL_TYPE(annotated_arg) annotated_arg<T *, detail::properties_t<Props...>> {
  using property_list_t = detail::properties_t<Props...>;

#ifdef __SYCL_DEVICE_ONLY__
  using global_pointer_t = typename decorated_global_ptr<T>::pointer;
#else
  using global_pointer_t = T *;
#endif

  global_pointer_t obj;

  template <typename T2, typename PropertyListT> friend class annotated_arg;

#ifdef __SYCL_DEVICE_ONLY__
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(
      detail::PropertyMetaInfo<Props>::name...,
      detail::PropertyMetaInfo<Props>::value...)]] global_pointer_t _obj) {
    obj = _obj;
  }
#endif

public:
  static_assert(is_property_list<property_list_t>::value,
                "Property list is invalid.");

  annotated_arg() noexcept = default;
  annotated_arg(const annotated_arg &) = default;
  annotated_arg &operator=(annotated_arg &) = default;

  annotated_arg(T *_ptr,
                const property_list_t &PropList = properties{}) noexcept
      : obj(global_pointer_t(_ptr)) {}

  // Constructs an annotated_arg object from a raw pointer and variadic
  // properties. The new property set contains all properties of the input
  // variadic properties. The same property in `Props...` and
  // `PropertyValueTs...` must have the same property value.
  template <typename... PropertyValueTs>
  annotated_arg(T *_ptr, const PropertyValueTs &...props) noexcept
      : obj(global_pointer_t(_ptr)) {
    static_assert(
        std::is_same<
            property_list_t,
            detail::merged_properties_t<property_list_t,
                                        decltype(properties{props...})>>::value,
        "The property list must contain all properties of the input of the "
        "constructor");
  }

  // Constructs an annotated_arg object from another annotated_arg object.
  // The new property set contains all properties of the input
  // annotated_arg object. The same property in `Props...` and `PropertyList2`
  // must have the same property value.
  template <typename T2, typename PropertyList2>
  explicit annotated_arg(const annotated_arg<T2, PropertyList2> &other) noexcept
      : obj(other.obj) {
    static_assert(std::is_convertible<T2, T *>::value,
                  "The underlying data type of the input annotated_arg is not "
                  "compatible");

    static_assert(
        std::is_same<
            property_list_t,
            detail::merged_properties_t<property_list_t, PropertyList2>>::value,
        "The constructed annotated_arg type must contain all the properties of "
        "the input annotated_arg");
  }

  // Constructs an annotated_arg object from another annotated_arg object and a
  // property list. The new property set is the union of property lists
  // `PropertyListU` and `PropertyListV`. The same property in `PropertyListU`
  // and `PropertyListV` must have the same property value.
  template <typename T2, typename PropertyListU, typename PropertyListV>
  explicit annotated_arg(const annotated_arg<T2, PropertyListU> &other,
                         const PropertyListV &proplist) noexcept
      : obj(other.obj) {
    static_assert(std::is_convertible<T2, T *>::value,
                  "The underlying data type of the input annotated_arg is not "
                  "compatible");

    static_assert(
        std::is_same<property_list_t, detail::merged_properties_t<
                                          PropertyListU, PropertyListV>>::value,
        "The property list of constructed annotated_arg type must be the union "
        "of the input property lists");
  }

  operator T *() noexcept { return obj; }
  operator T *() const noexcept { return obj; }

  T &operator[](std::ptrdiff_t idx) const noexcept { return obj[idx]; }

  template <typename PropertyT> static constexpr bool has_property() {
    return property_list_t::template has_property<PropertyT>();
  }

  template <typename PropertyT> static constexpr auto get_property() {
    return property_list_t::template get_property<PropertyT>();
  }
};

// Partial specialization for non-pointer type
template <typename T, typename... Props>
class __SYCL_SPECIAL_CLASS
__SYCL_TYPE(annotated_arg) annotated_arg<T, detail::properties_t<Props...>> {
  using property_list_t = detail::properties_t<Props...>;

  template <typename T2, typename PropertyListT> friend class annotated_arg;

  T obj;

#ifdef __SYCL_DEVICE_ONLY__
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(
      detail::PropertyMetaInfo<Props>::name...,
      detail::PropertyMetaInfo<Props>::value...)]] T _obj) {
    obj = _obj;
  }
#endif

public:
  static_assert(is_device_copyable_v<T>, "Type T must be device copyable.");
  static_assert(is_property_list<property_list_t>::value,
                "Property list is invalid.");
  static_assert(check_property_list<T, Props...>::value,
                "The property list contains invalid property.");

  annotated_arg() noexcept = default;
  annotated_arg(const annotated_arg &) = default;
  annotated_arg &operator=(annotated_arg &) = default;

  annotated_arg(const T &_obj,
                const property_list_t &PropList = properties{}) noexcept
      : obj(_obj) {}

  // Constructs an annotated_arg object from a raw pointer and variadic
  // properties. The new property set contains all properties of the input
  // variadic properties. The same property in `Props...` and
  // `PropertyValueTs...` must have the same property value.
  template <typename... PropertyValueTs>
  annotated_arg(const T &_obj, PropertyValueTs... props) noexcept : obj(_obj) {
    static_assert(
        std::is_same<
            property_list_t,
            detail::merged_properties_t<property_list_t,
                                        decltype(properties{props...})>>::value,
        "The property list must contain all properties of the input of the "
        "constructor");
  }

  // Constructs an annotated_arg object from another annotated_arg object.
  // The new property set contains all properties of the input
  // annotated_arg object. The same property in `Props...` and `PropertyList2`
  // must have the same property value.
  template <typename T2, typename PropertyList2>
  explicit annotated_arg(const annotated_arg<T2, PropertyList2> &other) noexcept
      : obj(other.obj) {
    static_assert(std::is_convertible<T2, T>::value,
                  "The underlying data type of the input annotated_arg is not "
                  "compatible");

    static_assert(
        std::is_same<
            property_list_t,
            detail::merged_properties_t<property_list_t, PropertyList2>>::value,
        "The constructed annotated_arg type must contain all the properties of "
        "the input annotated_arg");
  }

  // Constructs an annotated_arg object from another annotated_arg object and a
  // property list. The new property set is the union of property lists
  // `PropertyListU` and `PropertyListV`. The same property in `PropertyListU`
  // and `PropertyListV` must have the same property value.
  template <typename T2, typename PropertyListU, typename PropertyListV>
  explicit annotated_arg(const annotated_arg<T2, PropertyListU> &other,
                         const PropertyListV &proplist) noexcept
      : obj(other.obj) {
    static_assert(std::is_convertible<T2, T>::value,
                  "The underlying data type of the input annotated_arg is not "
                  "compatible");

    static_assert(
        std::is_same<property_list_t, detail::merged_properties_t<
                                          PropertyListU, PropertyListV>>::value,
        "The property list of constructed annotated_arg type must be the union "
        "of the input property lists");
  }

  operator T() noexcept { return obj; }
  operator T() const noexcept { return obj; }

  template <class RelayT = T>
  std::enable_if_t<detail::HasSubscriptOperator<RelayT>::value,
                   decltype(std::declval<RelayT>().operator[](0))> &
  operator[](std::ptrdiff_t idx) const noexcept {
    return obj.operator[](idx);
  }

  template <typename PropertyT> static constexpr bool has_property() {
    return property_list_t::template has_property<PropertyT>();
  }

  template <typename PropertyT> static constexpr auto get_property() {
    return property_list_t::template get_property<PropertyT>();
  }
};

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
