//==----------- graph_properties.hpp --- SYCL graph properties -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/property_helper.hpp>     // for DataLessPropKind
#include <sycl/properties/property_traits.hpp> // for is_property_of
#include <sycl/property_list.hpp>              // for property_list

#include <type_traits> // for true_type

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {
#define __SYCL_DATA_LESS_PROP(NS_QUALIFIER, PROP_NAME, ENUM_VAL)               \
  namespace NS_QUALIFIER {                                                     \
  class PROP_NAME                                                              \
      : public sycl::detail::DataLessProperty<sycl::detail::ENUM_VAL> {};      \
  }
#include <sycl/ext/oneapi/experimental/detail/properties/graph_properties.def>

#define __SYCL_DATA_LESS_PROP(NS_QUALIFIER, PROP_NAME, ENUM_VAL)               \
  namespace NS_QUALIFIER {                                                     \
  class PROP_NAME                                                              \
      : public sycl::detail::DataLessProperty<sycl::detail::ENUM_VAL> {};      \
  }
#include <sycl/ext/oneapi/experimental/detail/properties/node_properties.def>

class node;
namespace property::node {
class depends_on;
} // namespace property::node

enum class graph_state;
template <graph_state State> class command_graph;

namespace detail {
inline void checkGraphPropertiesAndThrow(const property_list &Properties) {
  auto CheckDataLessProperties = [](int PropertyKind) {
#define __SYCL_DATA_LESS_PROP(NS_QUALIFIER, PROP_NAME, ENUM_VAL)               \
  case NS_QUALIFIER::PROP_NAME::getKind():                                     \
    return true;
#define __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)
    switch (PropertyKind) {
#include <sycl/ext/oneapi/experimental/detail/properties/graph_properties.def>

    default:
      return false;
    }
  };
  // No properties with data for graph now.
  auto NoAllowedPropertiesCheck = [](int) { return false; };
  sycl::detail::PropertyValidator::checkPropsAndThrow(
      Properties, CheckDataLessProperties, NoAllowedPropertiesCheck);
}
} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext

#define __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)                  \
  template <ext::oneapi::experimental::graph_state State>                      \
  struct is_property_of<ext::oneapi::experimental::NS_QUALIFIER::PROP_NAME,    \
                        ext::oneapi::experimental::command_graph<State>>       \
      : std::true_type {};
#define __SYCL_DATA_LESS_PROP(NS_QUALIFIER, PROP_NAME, ENUM_VAL)               \
  __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)

#include <sycl/ext/oneapi/experimental/detail/properties/graph_properties.def>

#define __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)                  \
  template <>                                                                  \
  struct is_property_of<ext::oneapi::experimental::NS_QUALIFIER::PROP_NAME,    \
                        ext::oneapi::experimental::node> : std::true_type {};
#define __SYCL_DATA_LESS_PROP(NS_QUALIFIER, PROP_NAME, ENUM_VAL)               \
  __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)

#include <sycl/ext/oneapi/experimental/detail/properties/node_properties.def>

} // namespace _V1
} // namespace sycl
