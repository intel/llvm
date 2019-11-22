//==--------- property_list.hpp --- SYCL property list ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/property_base.hpp>
#include <CL/sycl/exception.hpp>
#include <memory>
#include <type_traits>
#include <typeindex>
#include <unordered_map>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

class property_list {

  // Base property wrapper type for type erasure
  class property_wrapper_base {
  public:
    virtual ~property_wrapper_base() = default;
  };

  // Polymorphic wrapper around a concrete property type
  template <typename T> class property_wrapper : public property_wrapper_base {
  public:
    T impl;

    property_wrapper(T impl) : impl{std::move(impl)} {}
  };

  // Map property type indicies against type erased container
  // Note: Memory fragmentation likely as each property is allocated seperately
  std::unordered_map<std::type_index, std::shared_ptr<property_wrapper_base>>
      properties;

  // Add a property to the map
  template <typename propertyT> void init_property_list(propertyT &&element) {
    using this_property = typename std::remove_reference<propertyT>::type;

    std::type_index index = std::type_index(typeid(this_property));

    properties[index] = std::make_shared<property_wrapper<this_property>>(
        std::forward<propertyT>(element));
  }

  // Recursively add each element in the parameter pack to the map
  template <typename propertyT, typename... propertyTN>
  void init_property_list(propertyT &&first, propertyTN &&... args) {
    init_property_list(std::forward<propertyT>(first));
    init_property_list(std::forward<propertyTN>(args)...);
  }

  template <typename Base, typename T, typename... Ts>
  struct are_base_of
      : std::conditional<std::is_base_of<Base, typename std::remove_reference<
                                                   T>::type>::value,
                         are_base_of<Base, Ts...>, std::false_type>::type {};

  template <typename Base, typename T>
  struct are_base_of<Base, T>
      : std::is_base_of<Base, typename std::remove_reference<T>::type> {};

public:
  property_list() {}

  // create a new property list from a parameter pack
  template <
      typename... propertyTN,
      typename = typename std::enable_if<
          are_base_of<property::detail::property_base, propertyTN...>::value,
          void>::type>
  property_list(propertyTN &&... args) {
    properties.reserve(sizeof...(propertyTN));

    init_property_list(std::forward<propertyTN>(args)...);
  }

  // check if a property exists in the property list
  template <typename propertyT> bool has_property() const {
    std::type_index index = std::type_index(typeid(propertyT));
    return properties.find(index) != properties.end();
  }

  // get a property in the property list
  template <typename propertyT> propertyT get_property() const {
    auto find = properties.find(std::type_index(typeid(propertyT)));

    if (find == properties.end() || find->second == nullptr) {
      throw invalid_object_error();
    }

    property_wrapper_base *base_prop_wrap = find->second.get();
    return static_cast<property_wrapper<propertyT> *>(base_prop_wrap)->impl;
  }
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
