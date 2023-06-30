//==-- annotated_ptr_specific_properties.hpp - SYCL properties specific for annotated_ptr --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {

template <typename T, typename PropertyListT> class annotated_ptr;

struct alignment_key {
  template <int K>
  using value_t = property_value<alignment_key, std::integral_constant<int, K>>;
};

template <int K> inline constexpr alignment_key::value_t<K> alignment;

template <> struct is_property_key<alignment_key> : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<alignment_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

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
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
