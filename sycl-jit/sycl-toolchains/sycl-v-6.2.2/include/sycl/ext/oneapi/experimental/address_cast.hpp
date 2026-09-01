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
namespace ext::oneapi ::experimental {
namespace detail {
using namespace sycl::detail;
}
// Shorthands for address space names
constexpr inline access::address_space global_space =
    access::address_space::global_space;
constexpr inline access::address_space local_space =
    access::address_space::local_space;
constexpr inline access::address_space private_space =
    access::address_space::private_space;
constexpr inline access::address_space generic_space =
    access::address_space::generic_space;

template <access::address_space Space, typename ElementType>
multi_ptr<ElementType, Space, access::decorated::no>
static_address_cast(ElementType *Ptr) {
  using ret_ty = multi_ptr<ElementType, Space, access::decorated::no>;
  return ret_ty{detail::static_address_cast<Space>(Ptr)};
}

template <access::address_space Space, access::decorated DecorateAddress,
          typename ElementType>
multi_ptr<ElementType, Space, DecorateAddress> static_address_cast(
    multi_ptr<ElementType, generic_space, DecorateAddress> Ptr) {
  if constexpr (Space == generic_space)
    return Ptr;
  else
    return {static_address_cast<Space>(Ptr.get_decorated())};
}

template <access::address_space Space, typename ElementType>
multi_ptr<ElementType, Space, access::decorated::no>
dynamic_address_cast(ElementType *Ptr) {
  using ret_ty = multi_ptr<ElementType, Space, access::decorated::no>;
  return ret_ty{detail::dynamic_address_cast<Space>(Ptr)};
}

template <access::address_space Space, access::decorated DecorateAddress,
          typename ElementType>
multi_ptr<ElementType, Space, DecorateAddress> dynamic_address_cast(
    multi_ptr<ElementType, generic_space, DecorateAddress> Ptr) {
  if constexpr (Space == generic_space)
    return Ptr;
  else
    return {dynamic_address_cast<Space>(Ptr.get_decorated())};
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
