//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>

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
} // namespace _V1
} // namespace sycl
