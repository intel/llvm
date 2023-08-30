//==----------- fpga_mem.hpp - SYCL fpga_mem extension -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>                       // for address_space
#include <sycl/exception.hpp>                           // for make_error_code
#include <sycl/ext/intel/experimental/fpga_mem/properties.hpp> // for num_banks
#include <sycl/ext/oneapi/properties/properties.hpp>    // for properties_t
// #include <sycl/pointers.hpp>                         // for decorated_gl...

#include <cstddef>     // for ptrdiff_t
#include <type_traits> // for enable_if_t
#include <utility>     // for declval


namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental {
namespace detail {

// Hide the base implementation in details so that manipulation
// of properties parameter pack can be modularized away from main logic
// template <typename T, typename PropertyListT, typename = void>
// class fpga_mem_base {
template <typename T, typename... Props>
class fpga_mem_base {

protected:
  T val
#ifdef __SYCL_DEVICE_ONLY__
      // Artem: Doesn't work for some reason
      [[__sycl_detail__::add_ir_annotations_member(
          ext::oneapi::experimental::detail::PropertyMetaInfo<Props>::name...,
          ext::oneapi::experimental::detail::PropertyMetaInfo<Props>::value...)]]
#endif
      ;

  T *get_ptr() noexcept {
    return &val; 
  }
  const T *get_ptr() const noexcept { 
    return &val;
  }

public:
  using element_type = std::remove_extent_t<T>;

  // All the initialization
  // constexpr is used as a hint to the compiler to try and evaluate the 
  // constructor at compile-time
  template<typename ... S>
  constexpr fpga_mem_base(S ... args) : val{args...} {}

  //Note: copy and move semantics should work for fpga_mem

  T &get() noexcept {
    return *this->get_ptr();
  }

  const T &get() const noexcept {
    return *this->get_ptr();
  }

  // Allows for implicit conversion from this to T
  operator T &() noexcept {
    return get();
  }

  // Allows for implicit conversion from this to T
  operator const T &() const noexcept {
    return get();
  }

  fpga_mem_base &operator=(const T &newValue) noexcept {
    *this->get_ptr() = newValue;
    return *this;
  }

  // Note that there is no need for "fpga_mem_base" to define member functions
  // for operators like "++", "[]", "->", comparison, etc. Instead, the type
  // "T" need only define these operators as non-member functions. Because
  // there is an implicit conversion from "fpga_mem_base" to "T&".
};
} // namespace detail

// Empty property list specialization
template <typename T, typename PropertyListT = ext::oneapi::experimental::detail::empty_properties_t>
class fpga_mem
    : public detail::fpga_mem_base<T> {

  // Inherits the base class' constructors
  using detail::fpga_mem_base<T>::fpga_mem_base;

public:
    template <typename propertyT> static constexpr bool has_property() {
    return false;
  }

  template <typename propertyT> static constexpr auto get_property() {
    return false;
  }
};

// alias for proper namespace
template <typename... Props>
using properties_t = ext::oneapi::experimental::detail::properties_t<Props...>;

template <typename T, typename... Props>
class fpga_mem<T, properties_t<Props...>>
    : public detail::fpga_mem_base<T, Props...> {

  using property_list_t = properties_t<Props...>;

    // Inherits the base class' constructors
  using detail::fpga_mem_base<T, Props...>::fpga_mem_base;

public:

  static_assert(ext::oneapi::experimental::is_property_list<property_list_t>::value,
                "Property list is invalid.");

    template <typename propertyT> static constexpr bool has_property() {
    return property_list_t::template has_property<propertyT>();
  }

  template <typename propertyT> static constexpr auto get_property() {
    return property_list_t::template get_property<propertyT>();
  }

};

} // namespace ext::intel::experimental
} // namespace _V1
} // namespace sycl

#undef __SYCL_HOST_NOT_SUPPORTED
