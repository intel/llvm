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


template <typename T, typename PropertyListT = detail::empty_properties_t, typename Enable = void>
class annotated_arg {
  // This should always fail when instantiating the unspecialized version.
  static_assert(is_property_list<PropertyListT>::value,
                "Property list is invalid.");
};

// Partial specialization for pointer type
template <typename T, typename... Props>
class __SYCL_SPECIAL_CLASS annotated_arg<T, detail::properties_t<Props...>, typename std::enable_if<std::is_pointer<T>::value>::type> {
  using property_list_t = detail::properties_t<Props...>;
  using UnderlyingT = typename std::remove_pointer<T>::type;
  __OPENCL_GLOBAL_AS__ UnderlyingT *ptr;

  #ifdef __SYCL_DEVICE_ONLY__
    void __init(
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          detail::PropertyMetaInfo<Props>::name...,
          detail::PropertyMetaInfo<Props>::value...
      )]]
      __OPENCL_GLOBAL_AS__ UnderlyingT* _ptr) {
        ptr = _ptr;
    }
  #endif

public:
  static_assert(std::is_trivially_destructible<T>::value,
                "Type T must be trivially destructible.");
  static_assert(is_property_list<property_list_t>::value,
                "Property list is invalid.");

  annotated_arg() = default;
  // annotated_arg(const annotated_arg&) = default;
  annotated_arg(UnderlyingT *_ptr) : ptr((__OPENCL_GLOBAL_AS__ UnderlyingT*)_ptr) {};

  operator T&() {
    __SYCL_HOST_NOT_SUPPORTED("Implicit conversion of annotated_arg to T")
    return ptr;
  }
  operator const T&() const {
    __SYCL_HOST_NOT_SUPPORTED("Implicit conversion of annotated_arg to T")
    return ptr;
  }

  // template<typename RelayT = T, typename = std::enable_if_t<std::is_pointer<RelayT>::value>>
  // std::remove_pointer_t<RelayT> operator [](std::ptrdiff_t idx) {
  //   __SYCL_HOST_NOT_SUPPORTED("operator[] on an annotated_arg")
  //   return ptr[idx];
  // }

  // auto operator [](std::ptrdiff_t idx) {
  //   __SYCL_HOST_NOT_SUPPORTED("operator[] on an annotated_arg")
  //   return ptr[idx];
  // }

  // inline T& get() {
  //   __SYCL_HOST_NOT_SUPPORTED("get()")
  //   return ptr;
  // }
  // inline const T& get() const {
  //   __SYCL_HOST_NOT_SUPPORTED("get()")
  //   return ptr;
  // }

  inline T get() const {
    __SYCL_HOST_NOT_SUPPORTED("get()")
    return ptr;
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
class __SYCL_SPECIAL_CLASS annotated_arg <T, detail::properties_t<Props...>, typename std::enable_if<!std::is_pointer<T>::value>::type> {
  using property_list_t = detail::properties_t<Props...>;

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
  static_assert(std::is_trivially_copy_constructible<T>::value,
                "Type T must be trivially copy constructable.");
  static_assert(std::is_trivially_destructible<T>::value,
                "Type T must be trivially destructible.");
  static_assert(is_property_list<property_list_t>::value,
                "Property list is invalid.");

  annotated_arg() = default;
  annotated_arg(const annotated_arg&) = default;
  annotated_arg(const T &_obj) : obj(_obj) {};

  operator T&() {
    __SYCL_HOST_NOT_SUPPORTED("Implicit conversion of annotated_arg to T")
    return obj;
  }
  operator const T&() const {
    __SYCL_HOST_NOT_SUPPORTED("Implicit conversion of annotated_arg to T")
    return obj;
  }

  inline T& get() {
    __SYCL_HOST_NOT_SUPPORTED("get()")
    return obj;
  }
  inline const T& get() const {
    __SYCL_HOST_NOT_SUPPORTED("get()")
    return obj;
  }

  template <typename PropertyT> static constexpr bool has_property() {
    return property_list_t::template has_property<PropertyT>();
  }

  template <typename PropertyT> static constexpr auto get_property() {
    return property_list_t::template get_property<PropertyT>();
  }
};


/*
template <typename T, typename PropertyListT = detail::empty_properties_t>
class annotated_arg {
  // This should always fail when instantiating the unspecialized version.
  static_assert(is_property_list<PropertyListT>::value,
                "Property list is invalid.");
};

// Partial specialization to make PropertyListT visible as a parameter pack
// of properties.
template <typename T, typename... Props>
class __SYCL_SPECIAL_CLASS annotated_arg<T, detail::properties_t<Props...>> {
  using property_list_t = detail::properties_t<Props...>;
  // using CondT = std::conditional<std::is_pointer<T>::value, __OPENCL_GLOBAL_AS__ UnderlyingT, T>::type;

  #ifdef __SYCL_DEVICE_ONLY__
    void __init(
      [[__sycl_detail__::add_ir_attributes_kernel_parameter(
          detail::PropertyMetaInfo<Props>::name...,
          detail::PropertyMetaInfo<Props>::value...
      )]]
      __OPENCL_GLOBAL_AS__ T* _obj) {
        obj = _obj;
    }
  #endif

public:
  // T should be trivially copy constructible to be device copyable
  static_assert(std::is_trivially_copy_constructible<T>::value,
                "Type T must be trivially copy constructable.");
  static_assert(std::is_trivially_destructible<T>::value,
                "Type T must be trivially destructible.");
  static_assert(is_property_list<property_list_t>::value,
                "Property list is invalid.");

  // Check compability of each property values in the property list
  // static_assert(check_property_list<T, Props...>::value,
  //               "property list contains invalid property.");

  annotated_arg() = default;
  annotated_arg(const annotated_arg&) = default;
  // annotated_arg(const __OPENCL_GLOBAL_AS__ T &_obj) : obj(_obj) {};
  annotated_arg(T *_obj) : obj((__OPENCL_GLOBAL_AS__ T*)_obj) {};

  operator T&() {
    __SYCL_HOST_NOT_SUPPORTED("Implicit conversion of annotated_arg to T")
    return obj;
  }
  operator const T&() const {
    __SYCL_HOST_NOT_SUPPORTED("Implicit conversion of annotated_arg to T")
    return obj;
  }

  // template<typename RelayT = T, typename = std::enable_if_t<std::is_pointer<RelayT>::value>>
  // std::remove_pointer_t<RelayT> operator [](std::ptrdiff_t idx) {
  //   __SYCL_HOST_NOT_SUPPORTED("operator[] on an annotated_arg")
  //   return obj[idx];
  // }

  // auto operator [](std::ptrdiff_t idx) {
  //   __SYCL_HOST_NOT_SUPPORTED("operator[] on an annotated_arg")
  //   return obj[idx];
  // }

  // inline T& get() {
  //   __SYCL_HOST_NOT_SUPPORTED("get()")
  //   return obj;
  // }
  // inline const T& get() const {
  //   __SYCL_HOST_NOT_SUPPORTED("get()")
  //   return obj;
  // }

  inline T* get() const {
    __SYCL_HOST_NOT_SUPPORTED("get()")
    return obj;
  }
  // inline const T* get() const {
  //   __SYCL_HOST_NOT_SUPPORTED("get()")
  //   return obj;
  // }

  template <typename PropertyT> static constexpr bool has_property() {
    return property_list_t::template has_property<PropertyT>();
  }

  template <typename PropertyT> static constexpr auto get_property() {
    return property_list_t::template get_property<PropertyT>();
  }
};
*/

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

#undef __SYCL_HOST_NOT_SUPPORTED
