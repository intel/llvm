//==-- properties.hpp - SYCL properties associated with
// annotated_arg/ptr --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/properties.hpp> // for properties_t

#include <type_traits> // for false_type, con...
#include <utility>     // for declval

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {

template <typename T, typename PropertyListT> class annotated_arg;
template <typename T, typename PropertyListT> class annotated_ptr;

//===----------------------------------------------------------------------===//
//        Common properties of annotated_arg/annotated_ptr
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//   Utility type trait for annotated_arg/annotated_ptr deduction guide
//===----------------------------------------------------------------------===//
template <typename T, typename PropertyValueT>
struct is_valid_property : std::false_type {};

namespace detail {
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

template <typename T, typename... Props>
struct check_property_list : std::true_type {};

template <typename T, typename Prop, typename... Props>
struct check_property_list<T, Prop, Props...>
    : std::conditional_t<is_valid_property<T, Prop>::value,
                         check_property_list<T, Props...>, std::false_type> {
  static_assert(is_valid_property<T, Prop>::value,
                "Property is invalid for the given type.");
};

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
