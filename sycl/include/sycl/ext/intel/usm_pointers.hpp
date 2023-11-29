//==-------- usm_pointers.hpp - Extended SYCL pointers classes -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp> // for decorated, address_space

namespace sycl {
inline namespace _V1 {

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
class multi_ptr;

namespace ext {
namespace intel {

template <typename ElementType,
          access::decorated IsDecorated = access::decorated::legacy>
using device_ptr =
    multi_ptr<ElementType, access::address_space::ext_intel_global_device_space,
              IsDecorated>;

template <typename ElementType,
          access::decorated IsDecorated = access::decorated::legacy>
using host_ptr =
    multi_ptr<ElementType, access::address_space::ext_intel_global_host_space,
              IsDecorated>;

// Template specialization aliases for different pointer address spaces.
// The interface exposes non-decorated pointer while keeping the
// address space information internally.

template <typename ElementType>
using raw_device_ptr =
    multi_ptr<ElementType, access::address_space::ext_intel_global_device_space,
              access::decorated::no>;

template <typename ElementType>
using raw_host_ptr =
    multi_ptr<ElementType, access::address_space::ext_intel_global_host_space,
              access::decorated::no>;

// Template specialization aliases for different pointer address spaces.
// The interface exposes decorated pointer.

template <typename ElementType>
using decorated_device_ptr =
    multi_ptr<ElementType, access::address_space::ext_intel_global_device_space,
              access::decorated::yes>;

template <typename ElementType>
using decorated_host_ptr =
    multi_ptr<ElementType, access::address_space::ext_intel_global_host_space,
              access::decorated::yes>;

} // namespace intel
} // namespace ext
} // namespace _V1
} // namespace sycl
