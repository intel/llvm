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
  static constexpr bool is_valid_property_for_given_type =
      is_valid_property<T, Prop>::value;
  static_assert(is_valid_property_for_given_type,
                "Property is invalid for the given type.");
};

template <typename PropTy> struct propagateToPtrAnnotation : std::false_type {};

// Partial specilization for property_value
template <typename PropKeyT, typename... PropValuesTs>
struct propagateToPtrAnnotation<property_value<PropKeyT, PropValuesTs...>>
    : propagateToPtrAnnotation<PropKeyT> {};

//===----------------------------------------------------------------------===//
//        Common properties of annotated_arg/annotated_ptr
//===----------------------------------------------------------------------===//
struct alignment_key {
  template <int K>
  using value_t = property_value<alignment_key, std::integral_constant<int, K>>;
};

template <int K> inline constexpr alignment_key::value_t<K> alignment;

template <> struct is_property_key<alignment_key> : std::true_type {};

template <typename T, int W>
struct is_valid_property<T, alignment_key::value_t<W>>
    : std::bool_constant<std::is_pointer<T>::value> {};

template <typename T, typename PropertyListT>
struct is_property_key_of<alignment_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<alignment_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};

template <> struct propagateToPtrAnnotation<alignment_key> : std::true_type {};

namespace detail {

template <> struct PropertyToKind<alignment_key> {
  static constexpr PropKind Kind = PropKind::Alignment;
};

template <> struct IsCompileTimeProperty<alignment_key> : std::true_type {};

template <int N> struct PropertyMetaInfo<alignment_key::value_t<N>> {
  static constexpr const char *name = "sycl-alignment";
  static constexpr int value = N;
};

} // namespace detail

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
