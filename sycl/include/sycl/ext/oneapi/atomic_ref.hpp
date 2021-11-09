//==----- atomic_ref.hpp - SYCL_ONEAPI_extended_atomics atomic_ref ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/atomic_ref.hpp>
#include <sycl/ext/oneapi/atomic_enums.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {

template <typename T, memory_order DefaultOrder, memory_scope DefaultScope,
          access::address_space AddressSpace>
using atomic_ref __SYCL2020_DEPRECATED("use 'sycl::atomic_ref' instead") =
    ::cl::sycl::atomic_ref<T, DefaultOrder, DefaultScope, AddressSpace>;

} // namespace oneapi
} // namespace ext

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
