//==-------------- memory_enums.hpp --- SYCL enums -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

enum class memory_order : int {
  relaxed = 0,
  acquire = 1,
  __consume_unsupported =
      2, // helps optimizer when mapping to std::memory_order
  release = 3,
  acq_rel = 4,
  seq_cst = 5
};

enum class memory_scope : int {
  work_item = 0,
  sub_group = 1,
  work_group = 2,
  device = 3,
  system = 4
};

#if __cplusplus >= 201703L
inline constexpr auto memory_scope_work_item = memory_scope::work_item;
inline constexpr auto memory_scope_sub_group = memory_scope::sub_group;
inline constexpr auto memory_scope_work_group = memory_scope::work_group;
inline constexpr auto memory_scope_device = memory_scope::device;
inline constexpr auto memory_scope_system = memory_scope::system;

inline constexpr auto memory_order_relaxed = memory_order::relaxed;
inline constexpr auto memory_order_acquire = memory_order::acquire;
inline constexpr auto memory_order_release = memory_order::release;
inline constexpr auto memory_order_acq_rel = memory_order::acq_rel;
inline constexpr auto memory_order_seq_cst = memory_order::seq_cst;
#endif

namespace detail {

inline std::vector<memory_order>
readMemoryOrderBitfield(pi_memory_order_capabilities bits) {
  std::vector<memory_order> result;
  if (bits & PI_MEMORY_ORDER_RELAXED)
    result.push_back(memory_order::relaxed);
  if (bits & PI_MEMORY_ORDER_ACQUIRE)
    result.push_back(memory_order::acquire);
  if (bits & PI_MEMORY_ORDER_RELEASE)
    result.push_back(memory_order::release);
  if (bits & PI_MEMORY_ORDER_ACQ_REL)
    result.push_back(memory_order::acq_rel);
  if (bits & PI_MEMORY_ORDER_SEQ_CST)
    result.push_back(memory_order::seq_cst);
  return result;
}

#ifndef __SYCL_DEVICE_ONLY__
static constexpr std::memory_order getStdMemoryOrder(sycl::memory_order order) {
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
#endif // __SYCL_DEVICE_ONLY__

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
