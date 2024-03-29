//==----------- address_cast.hpp - sycl_ext_oneapi_address_cast ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
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
  auto CastPtr = sycl::detail::spirv::GenericCastToPtr<Space>(Ptr);
  return multi_ptr<ElementType, Space, DecorateAddress>(CastPtr);
#else
  return multi_ptr<ElementType, Space, DecorateAddress>(Ptr);
#endif
}

template <access::address_space Space, access::decorated DecorateAddress,
          typename ElementType>
multi_ptr<ElementType, Space, DecorateAddress>
dynamic_address_cast(ElementType *Ptr) {
#ifdef __SYCL_DEVICE_ONLY__
  auto CastPtr = sycl::detail::spirv::GenericCastToPtrExplicit<Space>(Ptr);
  return multi_ptr<ElementType, Space, DecorateAddress>(CastPtr);
#else
  return multi_ptr<ElementType, Space, DecorateAddress>(Ptr);
#endif
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
