//==------------------ usm_impl.hpp - USM API Utils -------------*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/usm.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
namespace usm {

void *alignedAllocInternal(size_t Alignment, size_t Size,
                           const context_impl *CtxImpl,
                           const device_impl *DevImpl, sycl::usm::alloc Kind,
                           const property_list &PropList = {});

void freeInternal(void *Ptr, const context_impl *CtxImpl);

} // namespace usm
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
