//==--------- atomic_fence.hpp - SYCL 2020 atomic_fence --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/memory_enums.hpp> // for getStdMemoryOrder, memory_order

#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/detail/spirv.hpp>
#else
#include <atomic> // for atomic_thread_fence
#endif

namespace sycl {
inline namespace _V1 {

inline void atomic_fence(memory_order order, memory_scope scope) {
#ifdef __SYCL_DEVICE_ONLY__
  auto SPIRVOrder = detail::spirv::getMemorySemanticsMask(order);
  auto SPIRVScope = detail::spirv::getScope(scope);
  __spirv_MemoryBarrier(SPIRVScope, static_cast<uint32_t>(SPIRVOrder));
#else
  (void)scope;
  auto StdOrder = detail::getStdMemoryOrder(order);
  atomic_thread_fence(StdOrder);
#endif
}

} // namespace _V1
} // namespace sycl
