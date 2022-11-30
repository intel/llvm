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

#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_HOST_NOT_SUPPORTED(Op)
#else
#define __SYCL_HOST_NOT_SUPPORTED(Op)                                          \
  throw sycl::exception(                                                       \
      sycl::make_error_code(sycl::errc::feature_not_supported),                \
      Op " is not supported on host device.");
#endif

#ifdef __SYCL_DEVICE_ONLY__
#define __OPENCL_GLOBAL_AS__ __attribute__((opencl_global))
#ifdef __ENABLE_USM_ADDR_SPACE__
#define __OPENCL_GLOBAL_DEVICE_AS__ __attribute__((opencl_global_device))
#define __OPENCL_GLOBAL_HOST_AS__ __attribute__((opencl_global_host))
#else
#define __OPENCL_GLOBAL_DEVICE_AS__ __attribute__((opencl_global))
#define __OPENCL_GLOBAL_HOST_AS__ __attribute__((opencl_global))
#endif // __ENABLE_USM_ADDR_SPACE__
#define __OPENCL_LOCAL_AS__ __attribute__((opencl_local))
#define __OPENCL_CONSTANT_AS__ __attribute__((opencl_constant))
#define __OPENCL_PRIVATE_AS__ __attribute__((opencl_private))
#else
#define __OPENCL_GLOBAL_AS__
#define __OPENCL_GLOBAL_DEVICE_AS__
#define __OPENCL_GLOBAL_HOST_AS__
#define __OPENCL_LOCAL_AS__
#define __OPENCL_CONSTANT_AS__
#define __OPENCL_PRIVATE_AS__
#endif

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {

namespace detail {
// Type-trait for checking if a type defines `operator->`.
template <typename T, typename = void>
struct HasParenthesisOperator : std::false_type {};
template <typename T>
struct HasParenthesisOperator<
    T, sycl::detail::void_t<decltype(std::declval<T>().operator()())>>
    : std::true_type {};

template <typename T, typename = void>
struct HasSubscriptOperator : std::false_type {};

template <typename T>
struct HasSubscriptOperator<
    T, sycl::detail::void_t<decltype(std::declval<T>().operator[]())>>
    : std::true_type {};

} // namespace detail


template <typename T, typename... Args>
annotated_arg(T, Args... args) -> annotated_arg<T, detail::properties_t<Args...>, std::is_pointer<T>::value>;

// template <typename T, typename... Args>
// annotated_arg(T, properties<std::tuple<Args...>>) -> annotated_arg<T, detail::properties_t<Args...>, std::is_pointer<T>::value>;

template <typename T, typename old, typename ArgT, bool IsPtr>
annotated_arg(annotated_arg<T, old, IsPtr>, ArgT newp) -> annotated_arg<T, detail::merged_properties_t<old, ArgT>, IsPtr>;

template <typename T, typename PropertyListT = detail::empty_properties_t, bool IsPtr = std::is_pointer<T>::value>
class annotated_arg {
  // This should always fail when instantiating the unspecialized version.
  static_assert(is_property_list<PropertyListT>::value,
                "Property list is invalid.");
};

// Partial specialization for pointer type
template <typename T, typename... Props>
class __SYCL_SPECIAL_CLASS __SYCL_TYPE(annotated_arg) annotated_arg<T, detail::properties_t<Props...>, true> {
  using property_list_t = detail::properties_t<Props...>;
  using UnderlyingT = typename std::remove_pointer<T>::type;
  __OPENCL_GLOBAL_AS__ UnderlyingT *obj;

  template<typename T2, typename PropertyListT, bool OtherIsPtr>
  friend class annotated_arg;

  #ifdef __SYCL_DEVICE_ONLY__
    void __init(
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          detail::PropertyMetaInfo<Props>::name...,
          detail::PropertyMetaInfo<Props>::value...
      )]]
      __OPENCL_GLOBAL_AS__ UnderlyingT* _obj) {
        obj = _obj;
    }
  #endif

public:
  static_assert(std::is_trivially_destructible<T>::value,
                "Type T must be trivially destructible.");
  // static_assert(is_property_list<property_list_t>::value,
                // "Property list is invalid.");
  static_assert(check_property_list<T, Props...>::value,
                "The property list contains invalid property.");

  annotated_arg() noexcept = default;
  annotated_arg(const annotated_arg&) = default;
  annotated_arg& operator=(annotated_arg&) = default;

  // explicit annotated_arg(const T& _ptr, const property_list_t &PropList = properties{}) noexcept
  //   : obj((__OPENCL_GLOBAL_AS__ UnderlyingT*)_ptr) {}

  template<typename... PropertyValueTs>
  explicit annotated_arg(const T& _ptr, PropertyValueTs... props) noexcept : obj((__OPENCL_GLOBAL_AS__ UnderlyingT*)_ptr) {
    static_assert(
        std::is_same<
            property_list_t,
            detail::merged_properties_t<property_list_t, detail::properties_t<PropertyValueTs...>>>::value,
        "The property list must contain all properties of the input of the constructor"
    );
  }
  
  // Constructs an annotated_arg object from another annotated_arg object.
  // The property set PropertyListT contains all properties of the input annotated_arg object.
  // If there are duplicate properties present in the property list of the input annotated_arg object,
  // the values of the duplicate properties must be the same.
  template <typename T2, typename PropertyList2>
  explicit annotated_arg(const annotated_arg<T2, PropertyList2> &other) noexcept : obj(other.obj) {
    static_assert(std::is_convertible<T2, T>::value, 
      "The underlying data type of the input annotated_arg is not compatible");

    static_assert(
        std::is_same<
            property_list_t,
            detail::merged_properties_t<property_list_t, PropertyList2>>::value,
        "The constructed annotated_arg type must contain all the properties of the input annotated_arg");
  }

  template <typename T2, typename PropertyListU, typename PropertyListV>
  explicit annotated_arg(const annotated_arg<T2, PropertyListU> &other,
      const PropertyListV& proplist) noexcept : obj(other.obj) {
    static_assert(std::is_convertible<T2, T>::value, 
      "The underlying data type of the input annotated_arg is not compatible");

    static_assert(
        std::is_same<
            property_list_t,
            detail::merged_properties_t<PropertyListU, PropertyListV>>::value,
        "The property list of constructed annotated_arg type must be the union of the input property lists");
  }

  operator T() noexcept {
    return  obj;
  }

  operator const T() const noexcept {
    return obj;
  }

  UnderlyingT& operator [](std::ptrdiff_t idx) const noexcept {
    return obj[idx];
  }

  template <typename PropertyT> static constexpr bool has_property() {
    return property_list_t::template has_property<PropertyT>();
  }

  template <typename PropertyT> static constexpr auto get_property() {
    return property_list_t::template get_property<PropertyT>();
  }
};


// Partial specialization for non-pointer type
template <typename T, typename... Props>
class __SYCL_SPECIAL_CLASS __SYCL_TYPE(annotated_arg) annotated_arg <T, detail::properties_t<Props...>, false> {
  using property_list_t = detail::properties_t<Props...>;

  template<typename T2, typename PropertyListT, bool OtherIsPtr>
  friend class annotated_arg;

  T obj;

  #ifdef __SYCL_DEVICE_ONLY__
    void __init(
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          detail::PropertyMetaInfo<Props>::name...,
          detail::PropertyMetaInfo<Props>::value...
      )]]
      T _obj) {
        obj = _obj;
    }
  #endif

public:
  // T should be trivially copy constructible to be device copyable
  static_assert(std::is_trivially_copyable<T>::value,
                "Type T must be trivially copyable.");
  // static_assert(std::is_trivially_copy_constructible<T>::value,
  //               "Type T must be trivially copy constructable.");
  static_assert(std::is_trivially_destructible<T>::value,
                "Type T must be trivially destructible.");
  static_assert(is_property_list<property_list_t>::value,
                "Property list is invalid.");
  static_assert(check_property_list<T, Props...>::value,
                "The property list contains invalid property.");

  annotated_arg() noexcept = default;
  annotated_arg(const annotated_arg&) = default;
  annotated_arg& operator=(annotated_arg&) = default;

  explicit annotated_arg(const T& _obj, const property_list_t &PropList = properties{}) noexcept : obj(_obj) {}

  template<typename... PropertyValueTs>
  explicit annotated_arg(const T& _obj, PropertyValueTs... props) noexcept : obj(_obj) {
    static_assert(
        std::is_same<
            property_list_t,
            detail::merged_properties_t<property_list_t, detail::properties_t<PropertyValueTs...>>>::value,
        "The property list must contain all properties of the input of the constructor"
    );
  }

  // Constructs an annotated_arg object from another annotated_arg object.
  // The property set PropertyListT contains all properties of the input annotated_arg object.
  // If there are duplicate properties present in the property list of the input annotated_arg object,
  // the values of the duplicate properties must be the same.
  template <typename T2, typename PropertyList2>
  explicit annotated_arg(const annotated_arg<T2, PropertyList2> &other) noexcept : obj(other.obj) {
    static_assert(std::is_convertible<T2, T>::value,
      "The underlying data type of the input annotated_arg is not compatible");

    static_assert(
        std::is_same<
            property_list_t,
            detail::merged_properties_t<property_list_t, PropertyList2>>::value,
        "The constructed annotated_arg type must contain all the properties of the input annotated_arg");
  }

  template <typename T2, typename PropertyListU, typename PropertyListV>
  explicit annotated_arg(const annotated_arg<T2, PropertyListU> &other,
      const PropertyListV& proplist) noexcept : obj(other.obj) {
    static_assert(std::is_convertible<T2, T>::value,
      "The underlying data type of the input annotated_arg is not compatible");

     static_assert(
        std::is_same<
            property_list_t,
            detail::merged_properties_t<PropertyListU, PropertyListV>>::value,
        "The property list of constructed annotated_arg type must be the union of the input property lists");
  }

  operator T() noexcept {
    return obj;
  }
  operator const T() const noexcept {
    return obj;
  }

  // template<typename... Args>
  // template <class RelayT = T>
  // std::enable_if_t<detail::HasParenthesisOperator<RelayT>::value>
  //     &operator()(Args... args) noexcept {
  //   __SYCL_HOST_NOT_SUPPORTED("operator() on an annotated_arg")
  //   return obj.operator(args);
  // }

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

#undef __SYCL_HOST_NOT_SUPPORTED
