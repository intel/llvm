//==--- static_addrspace_cast.hpp --- KHR static addrspace cast extension --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#ifdef __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS

#include <sycl/access/access.hpp>
#include <sycl/ext/oneapi/experimental/address_cast.hpp>
#include <sycl/multi_ptr.hpp>

#define SYCL_KHR_STATIC_ADDRSPACE_CAST 1

namespace sycl {
inline namespace _V1 {
namespace khr {

template <access::address_space Space, typename ElementType>
multi_ptr<ElementType, Space, access::decorated::no>
static_addrspace_cast(ElementType *ptr) {
  return ext::oneapi::experimental::static_address_cast<Space>(ptr);
}

template <access::address_space Space, typename ElementType,
          access::decorated DecorateAddress>
multi_ptr<ElementType, Space, DecorateAddress> static_addrspace_cast(
    multi_ptr<ElementType, access::address_space::generic_space,
              DecorateAddress>
        ptr) {
  return ext::oneapi::experimental::static_address_cast<Space>(ptr);
}

} // namespace khr
} // namespace _V1
} // namespace sycl

#endif
