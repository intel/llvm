//==----------- annotated_ptr.hpp - SYCL annotated_ptr extension -----------==//
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
#include <sycl/ext/oneapi/properties/properties.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {

#define PROPAGATE_OP(op)                                                       \
annotated_ref operator op(const T &rhs) noexcept {                           \
  (*ptr) op rhs;                                                             \
  return *this;                                                              \
}

template <typename T, typename PropertyListT = detail::empty_properties_t>
class annotated_ref {
  // This should always fail when instantiating the unspecialized version.
  static_assert(is_property_list<PropertyListT>::value,
                "Property list is invalid.");
};

template <typename T, typename... Props>
class annotated_ref<T, detail::properties_t<Props...>> {
  using property_list_t = detail::properties_t<Props...>;

private:
  T *ptr
#ifdef __SYCL_DEVICE_ONLY__
      [[__sycl_detail__::add_ir_annotations_member(
          detail::PropertyMetaInfo<Props>::name...,
          detail::PropertyMetaInfo<Props>::value...)]]
#endif
      ;

public:
  annotated_ref(T *_ptr) : ptr(_ptr) {}
  annotated_ref(const annotated_ref &) = default;

  operator T() const { return *ptr; }

  annotated_ref &operator=(const T &obj) {
    *ptr = obj;
    return *this;
  }

  // annotated_ref& operator=(const annotated_ref&) const = default;
  annotated_ref &operator=(const annotated_ref &) = default;

  PROPAGATE_OP(+=)
  PROPAGATE_OP(-=)
  PROPAGATE_OP(*=)
  PROPAGATE_OP(/=)
  PROPAGATE_OP(%=)
  PROPAGATE_OP(^=)
  PROPAGATE_OP(&=)
  PROPAGATE_OP(|=)
};

#undef PROPAGATE_OP
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
