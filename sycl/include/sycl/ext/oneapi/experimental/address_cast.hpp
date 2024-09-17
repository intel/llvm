//==----------- address_cast.hpp - sycl_ext_oneapi_address_cast ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/spirv.hpp>
#include <sycl/multi_ptr.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {

template <access::address_space Space, access::decorated DecorateAddress,
          typename ElementType>
multi_ptr<ElementType, Space, DecorateAddress>
static_address_cast(ElementType *Ptr) {
#ifdef __SYCL_DEVICE_ONLY__
  // TODO: Remove this restriction.
  static_assert(std::is_same_v<ElementType, remove_decoration_t<ElementType>>,
                "The extension expect undecorated raw pointers only!");
  if constexpr (Space == access::address_space::generic_space) {
    // Undecorated raw pointer is in generic AS already, no extra casts needed.
    // Note for future, for `OpPtrCastToGeneric`, `Pointer` must point to one of
    // `Storage Classes` that doesn't include `Generic`, so this will have to
    // remain a special case even if the restriction above is lifted.
    return multi_ptr<ElementType, Space, DecorateAddress>(Ptr);
  } else {
    auto CastPtr = sycl::detail::spirv::GenericCastToPtr<Space>(Ptr);
    return multi_ptr<ElementType, Space, DecorateAddress>(CastPtr);
  }
#else
  return multi_ptr<ElementType, Space, DecorateAddress>(Ptr);
#endif
}

template <access::address_space Space, access::decorated DecorateAddress,
          typename ElementType>
multi_ptr<ElementType, Space, DecorateAddress>
dynamic_address_cast(ElementType *Ptr) {
#ifdef __SYCL_DEVICE_ONLY__
  // TODO: Remove this restriction.
  static_assert(std::is_same_v<ElementType, remove_decoration_t<ElementType>>,
                "The extension expect undecorated raw pointers only!");
  if constexpr (Space == access::address_space::generic_space) {
    return multi_ptr<ElementType, Space, DecorateAddress>(Ptr);
  } else {
    auto CastPtr = sycl::detail::spirv::GenericCastToPtrExplicit<Space>(Ptr);
    return multi_ptr<ElementType, Space, DecorateAddress>(CastPtr);
  }
#else
  return multi_ptr<ElementType, Space, DecorateAddress>(Ptr);
#endif
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
