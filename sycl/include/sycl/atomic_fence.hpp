//==--------- atomic_fence.hpp - SYCL 2020 atomic_fence --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <sycl/detail/spirv.hpp>
#include <sycl/memory_enums.hpp>

#ifndef __SYCL_DEVICE_ONLY__
#include <atomic>
#endif

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

static inline void atomic_fence(memory_order order, memory_scope scope) {
#ifdef __SYCL_DEVICE_ONLY__
  constexpr auto SPIRVOrder = detail::spirv::getMemorySemanticsMask(order);
  constexpr auto SPIRVScope = detail::spirv::getScope(scope);
  __spirv_MemoryBarrier(SPIRVScope, static_cast<uint32_t>(SPIRVOrder));
#else
  (void)scope;
  auto StdOrder = detail::getStdMemoryOrder(order);
  atomic_thread_fence(StdOrder);
#endif
}

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
