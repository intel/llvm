//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>

#include <type_traits>

namespace sycl {
inline namespace _V1 {
// Forward declaration
template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
class multi_ptr;
template <access::address_space Space, access::decorated DecorateAddress,
          typename ElementType>
multi_ptr<ElementType, Space, DecorateAddress>
address_space_cast(ElementType *pointer);

namespace detail {
template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
std::add_pointer_t<ElementType>
get_raw_pointer(multi_ptr<ElementType, Space, DecorateAddress> Ptr);

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
auto get_element_pointer(multi_ptr<ElementType, Space, DecorateAddress> Ptr);
} // namespace detail
} // namespace _V1
} // namespace sycl
