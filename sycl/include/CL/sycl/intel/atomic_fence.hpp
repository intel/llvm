//==----- atomic_fence.hpp - SYCL_INTEL_extended_atomics atomic_fence ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/sycl/detail/spirv.hpp>
#include <CL/sycl/intel/atomic_enums.hpp>

#ifndef __SYCL_DEVICE_ONLY__
#include <atomic>
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace intel {
namespace detail {
using namespace cl::sycl::detail;
}

static inline void atomic_fence(memory_order order, memory_scope scope) {
#ifdef __SYCL_DEVICE_ONLY__
  auto spirv_order = detail::spirv::getMemorySemanticsMask(order);
  auto spirv_scope = detail::spirv::getScope(scope);
  __spirv_MemoryBarrier(spirv_scope, static_cast<uint32_t>(spirv_order));
#else
  (void)scope;
  auto std_order = detail::getStdMemoryOrder(order);
  atomic_thread_fence(std_order);
#endif
}

} // namespace intel
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
