//==----------- fpga_mem.hpp - SYCL fpga_mem extension -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp> // for address_space
#include <sycl/exception.hpp>     // for make_error_code
#include <sycl/ext/intel/experimental/fpga_mem/properties.hpp> // for num_banks
#include <sycl/ext/oneapi/properties/properties.hpp> // for properties_t

#include <cstddef>     // for ptrdiff_t
#include <type_traits> // for enable_if_t
#include <utility>     // for declval

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental {

// Primary template should never be deduced. Needed to establish which
// parameters fpga_mem can be templated on
template <typename T, typename PropertyListT =
                          ext::oneapi::experimental::empty_properties_t>
class fpga_mem {

  static_assert(
      ext::oneapi::experimental::is_property_list<PropertyListT>::value,
      "Property list is invalid.");
};

// Template specialization that all calls should use. Separates Props from
// properties_t which allows the class to apply Props as attributes.
template <typename T, typename... Props>
class
#ifdef __SYCL_DEVICE_ONLY__
    [[__sycl_detail__::add_ir_attributes_global_variable(
        "sycl-resource",
        ext::oneapi::experimental::detail::PropertyMetaInfo<Props>::name...,
        "DEFAULT",
        ext::oneapi::experimental::detail::PropertyMetaInfo<Props>::value...)]]
#endif
    fpga_mem<T, ext::oneapi::experimental::detail::properties_t<Props...>> {

protected:
  T val
#ifdef __SYCL_DEVICE_ONLY__
      // In addition to annotating all the user specified properties, also add
      // {"sycl-resource", ""} hinting to the optimizer that this object
      // should be implemented in memory outside the datapath.
      [[__sycl_detail__::add_ir_annotations_member(
          "sycl-resource",
          ext::oneapi::experimental::detail::PropertyMetaInfo<Props>::name...,
          "DEFAULT",
          ext::oneapi::experimental::detail::PropertyMetaInfo<
              Props>::value...)]]
#endif
      ;

public:
  using element_type = std::remove_extent_t<T>;

  // All the initialization
  // constexpr is used as a hint to the compiler to try and evaluate the
  // constructor at compile-time
  template <typename... S> constexpr fpga_mem(S... args) : val{args...} {}

  fpga_mem() = default;

  fpga_mem(const fpga_mem &) = default;
  fpga_mem(fpga_mem &&) = default;
  fpga_mem &operator=(const fpga_mem &) = default;
  fpga_mem &operator=(fpga_mem &&) = default;

  T &get() noexcept { return val; }

  constexpr const T &get() const noexcept { return val; }

  // Allows for implicit conversion from this to T
  operator T &() noexcept { return get(); }

  // Allows for implicit conversion from this to T
  constexpr operator const T &() const noexcept { return get(); }

  fpga_mem &operator=(const T &newValue) noexcept {
    val = newValue;
    return *this;
  }

  // Note that there is no need for "fpga_mem" to define member functions
  // for operators like "++", "[]", "->", comparison, etc. Instead, the type
  // "T" need only define these operators as non-member functions. Because
  // there is an implicit conversion from "fpga_mem" to "T&".

  template <typename propertyT> static constexpr bool has_property() {
    return ext::oneapi::experimental::detail::properties_t<
        Props...>::template has_property<propertyT>();
  }

  template <typename propertyT> static constexpr auto get_property() {
    return ext::oneapi::experimental::detail::properties_t<
        Props...>::template get_property<propertyT>();
  }
};

} // namespace ext::intel::experimental
} // namespace _V1
} // namespace sycl
