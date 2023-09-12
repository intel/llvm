//==----------- fpga_datapath.hpp - SYCL fpga_datapath extension -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>                       // for address_space
#include <sycl/exception.hpp>                           // for make_error_code
#include <sycl/ext/intel/experimental/fpga_mem/fpga_mem.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>    // for properties_t
#include <sycl/ext/oneapi/properties/property.hpp>       // for PropKind
#include <sycl/ext/oneapi/properties/property_value.hpp> // for property_value

#include <cstddef>     // for ptrdiff_t
#include <type_traits> // for enable_if_t
#include <utility>     // for declval


namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental {
  // shorthand used with properties
  namespace oneapi_exp = ext::oneapi::experimental;
namespace detail {

// Non-user accessible or documented property. Internal use only.
// Implementation detail used in-order to limit 
// duplication of code between fpga_mem and fpga_datapath.
struct datapath_key {
  using value_t = oneapi_exp::property_value<datapath_key>;
};

} // namespace detail
} // namespace ext::intel::experimental

namespace ext::oneapi::experimental::detail {
// Note: Property 'datapath' is not a user accessible property. It is only used 
// to generate appropriate IR pointer annotations on the fpga_datapath object.
// Therefore it doesn't need to override user quires, ie. is_property_key_of. 

using datapath_key = ext::intel::experimental::detail::datapath_key;

// Map Property to a PropKind enum
template <> struct PropertyToKind<datapath_key> {
  static constexpr PropKind Kind = PropKind::Datapath;
};

template <> struct IsCompileTimeProperty<datapath_key> : std::true_type {};

template <> struct PropertyMetaInfo<datapath_key::value_t> {
  static constexpr const char *name = "sycl-datapath";
  static constexpr std::nullptr_t value = nullptr;
};

} // namespace ext::oneapi::experimental::detail

namespace ext::intel::experimental {
// alias for proper namespace
template <typename... Props>
using properties_t = oneapi_exp::detail::properties_t<Props...>;

template <typename T>
class 
#ifdef __SYCL_DEVICE_ONLY__
      [[__sycl_detail__::add_ir_attributes_global_variable(
          oneapi_exp::detail::PropertyMetaInfo<detail::datapath_key::value_t>::name, 
          oneapi_exp::detail::PropertyMetaInfo<detail::datapath_key::value_t>::value)]]
#endif
fpga_datapath
    : public detail::fpga_mem_base<T, detail::datapath_key> {

  // Inherits the base class' constructors
  using detail::fpga_mem_base<T, detail::datapath_key>::fpga_mem_base;

};

} // namespace ext::intel::experimental

} // namespace _V1
} // namespace sycl

#undef __SYCL_HOST_NOT_SUPPORTED
