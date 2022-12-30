//==------------ pointers.hpp - SYCL pointers classes ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <sycl/access/access.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
class multi_ptr;
// Template specialization aliases for different pointer address spaces

template <typename ElementType,
          access::decorated IsDecorated = access::decorated::legacy>
using generic_ptr =
    multi_ptr<ElementType, access::address_space::generic_space, IsDecorated>;

template <typename ElementType,
          access::decorated IsDecorated = access::decorated::legacy>
using global_ptr =
    multi_ptr<ElementType, access::address_space::global_space, IsDecorated>;

// Note: Templated alias deprecation is not currently working in clang. See
// https://github.com/llvm/llvm-project/issues/18236.
template <typename ElementType,
          access::decorated IsDecorated = access::decorated::legacy>
using device_ptr __SYCL_DEPRECATED(
    "Use 'sycl::ext::intel::device_ptr' instead.") =
    multi_ptr<ElementType, access::address_space::ext_intel_global_device_space,
              IsDecorated>;

// Note: Templated alias deprecation is not currently working in clang. See
// https://github.com/llvm/llvm-project/issues/18236.
template <typename ElementType,
          access::decorated IsDecorated = access::decorated::legacy>
using host_ptr __SYCL_DEPRECATED("Use 'sycl::ext::intel::host_ptr' instead.") =
    multi_ptr<ElementType, access::address_space::ext_intel_global_host_space,
              IsDecorated>;

template <typename ElementType,
          access::decorated IsDecorated = access::decorated::legacy>
using local_ptr =
    multi_ptr<ElementType, access::address_space::local_space, IsDecorated>;

template <typename ElementType>
using constant_ptr =
    multi_ptr<ElementType, access::address_space::constant_space,
              access::decorated::legacy>;

template <typename ElementType,
          access::decorated IsDecorated = access::decorated::legacy>
using private_ptr =
    multi_ptr<ElementType, access::address_space::private_space, IsDecorated>;

// Template specialization aliases for different pointer address spaces.
// The interface exposes non-decorated pointer while keeping the
// address space information internally.

template <typename ElementType>
using raw_global_ptr =
    multi_ptr<ElementType, access::address_space::global_space,
              access::decorated::no>;

template <typename ElementType>
using raw_local_ptr = multi_ptr<ElementType, access::address_space::local_space,
                                access::decorated::no>;

template <typename ElementType>
using raw_private_ptr =
    multi_ptr<ElementType, access::address_space::private_space,
              access::decorated::no>;

// Template specialization aliases for different pointer address spaces.
// The interface exposes decorated pointer.

template <typename ElementType>
using decorated_global_ptr =
    multi_ptr<ElementType, access::address_space::global_space,
              access::decorated::yes>;

template <typename ElementType>
using decorated_local_ptr =
    multi_ptr<ElementType, access::address_space::local_space,
              access::decorated::yes>;

template <typename ElementType>
using decorated_private_ptr =
    multi_ptr<ElementType, access::address_space::private_space,
              access::decorated::yes>;

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
