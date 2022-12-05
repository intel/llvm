//==-- owner_less_base.hpp --- Common base class for owner-based ordering --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/common.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/ext/oneapi/weak_object_base.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

// Common CRTP base class supplying a common definition of owner-before ordering
// for SYCL objects.
template <class SyclObjT> class OwnerLessBase {
public:
#ifndef __SYCL_DEVICE_ONLY__
  /// Compares the object against a weak object using an owner-based
  /// implementation-defined ordering.
  ///
  /// \param Other is the weak object to compare ordering against.
  /// \return true if this object precedes \param Other and false otherwise.
  bool ext_oneapi_owner_before(
      const ext::oneapi::detail::weak_object_base<SyclObjT> &Other)
      const noexcept {
    return getSyclObjImpl(*static_cast<const SyclObjT *>(this))
        .owner_before(ext::oneapi::detail::getSyclWeakObjImpl(Other));
  }

  /// Compares the object against another object using an owner-based
  /// implementation-defined ordering.
  ///
  /// \param Other is the object to compare ordering against.
  /// \return true if this object precedes \param Other and false otherwise.
  bool ext_oneapi_owner_before(const SyclObjT &Other) const noexcept {
    return getSyclObjImpl(*static_cast<const SyclObjT *>(this))
        .owner_before(getSyclObjImpl(Other));
  }
#endif
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
