//==---- atomic_fence.hpp - SYCL_ONEAPI_extended_atomics atomic_fence ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/sycl/detail/spirv.hpp>
#include <sycl/ext/oneapi/atomic_enums.hpp>

#ifndef __SYCL_DEVICE_ONLY__
#include <atomic>
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace detail {
using namespace cl::sycl::detail;
}

__SYCL2020_DEPRECATED("use sycl::atomic_fence instead")
static inline void atomic_fence(memory_order order, memory_scope scope) {
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

} // namespace oneapi
} // namespace ext

namespace __SYCL2020_DEPRECATED("use 'ext::oneapi' instead") ONEAPI {
  using namespace ext::oneapi;
}
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
