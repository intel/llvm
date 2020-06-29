//==---------------- atomic_enums.hpp - SYCL_INTEL_extended_atomics enums --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/helpers.hpp>

#ifndef __SYCL_DEVICE_ONLY__
#include <atomic>
#endif
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace intel {

enum class memory_order : int {
  relaxed,
  acquire,
  __consume_unsupported, // helps optimizer when mapping to std::memory_order
  release,
  acq_rel,
  seq_cst
};
__SYCL_INLINE_CONSTEXPR memory_order memory_order_relaxed =
    memory_order::relaxed;
__SYCL_INLINE_CONSTEXPR memory_order memory_order_acquire =
    memory_order::acquire;
__SYCL_INLINE_CONSTEXPR memory_order memory_order_release =
    memory_order::release;
__SYCL_INLINE_CONSTEXPR memory_order memory_order_acq_rel =
    memory_order::acq_rel;
__SYCL_INLINE_CONSTEXPR memory_order memory_order_seq_cst =
    memory_order::seq_cst;

enum class memory_scope : int {
  work_item,
  sub_group,
  work_group,
  device,
  system
};
__SYCL_INLINE_CONSTEXPR memory_scope memory_scope_work_item =
    memory_scope::work_item;
__SYCL_INLINE_CONSTEXPR memory_scope memory_scope_sub_group =
    memory_scope::sub_group;
__SYCL_INLINE_CONSTEXPR memory_scope memory_scope_work_group =
    memory_scope::work_group;
__SYCL_INLINE_CONSTEXPR memory_scope memory_scope_device = memory_scope::device;
__SYCL_INLINE_CONSTEXPR memory_scope memory_scope_system = memory_scope::system;

#ifndef __SYCL_DEVICE_ONLY__
namespace detail {
// Cannot use switch statement in constexpr before C++14
// Nested ternary conditions in else branch required for C++11
#if __cplusplus >= 201402L
static inline constexpr std::memory_order
getStdMemoryOrder(::cl::sycl::intel::memory_order order) {
  switch (order) {
  case memory_order::relaxed:
    return std::memory_order_relaxed;
  case memory_order::__consume_unsupported:
    return std::memory_order_consume;
  case memory_order::acquire:
    return std::memory_order_acquire;
  case memory_order::release:
    return std::memory_order_release;
  case memory_order::acq_rel:
    return std::memory_order_acq_rel;
  case memory_order::seq_cst:
    return std::memory_order_seq_cst;
  }
}
#else
static inline constexpr std::memory_order
getStdMemoryOrder(::cl::sycl::intel::memory_order order) {
  return (order == memory_order::relaxed)
             ? std::memory_order_relaxed
             : (order == memory_order::__consume_unsupported)
                   ? std::memory_order_consume
                   : (order == memory_order::acquire)
                         ? std::memory_order_acquire
                         : (order == memory_order::release)
                               ? std::memory_order_release
                               : (order == memory_order::acq_rel)
                                     ? std::memory_order_acq_rel
                                     : std::memory_order_seq_cst;
}
#endif // __cplusplus
} // namespace detail
#endif // __SYCL_DEVICE_ONLY__

} // namespace intel
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
