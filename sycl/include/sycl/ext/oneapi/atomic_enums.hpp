//==--------------- atomic_enums.hpp - SYCL_ONEAPI_extended_atomics enums --==//
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
#include <CL/sycl/memory_enums.hpp>

#ifndef __SYCL_DEVICE_ONLY__
#include <atomic>
#endif
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {

using memory_order __SYCL2020_DEPRECATED("use 'sycl::memory_order' instead") =
    cl::sycl::memory_order;
__SYCL_INLINE_CONSTEXPR memory_order memory_order_relaxed __SYCL2020_DEPRECATED(
    "use 'sycl::memory_order_relaxed' instead") = memory_order::relaxed;
__SYCL_INLINE_CONSTEXPR memory_order memory_order_acquire __SYCL2020_DEPRECATED(
    "use 'sycl::memory_order_acquire' instead") = memory_order::acquire;
__SYCL_INLINE_CONSTEXPR memory_order memory_order_release __SYCL2020_DEPRECATED(
    "use 'sycl::memory_order_release' instead") = memory_order::release;
__SYCL_INLINE_CONSTEXPR memory_order memory_order_acq_rel __SYCL2020_DEPRECATED(
    "use 'sycl::memory_order_acq_rel' instead") = memory_order::acq_rel;
__SYCL_INLINE_CONSTEXPR memory_order memory_order_seq_cst __SYCL2020_DEPRECATED(
    "use 'sycl::memory_order_seq_cst' instead") = memory_order::seq_cst;

using memory_scope __SYCL2020_DEPRECATED("use 'sycl::memory_scope' instead") =
    cl::sycl::memory_scope;
__SYCL_INLINE_CONSTEXPR
memory_scope memory_scope_work_item __SYCL2020_DEPRECATED(
    "use 'sycl::memory_scope_work_item' instead") = memory_scope::work_item;
__SYCL_INLINE_CONSTEXPR
memory_scope memory_scope_sub_group __SYCL2020_DEPRECATED(
    "use 'sycl::memory_scope_sub_group' instead") = memory_scope::sub_group;
__SYCL_INLINE_CONSTEXPR memory_scope memory_scope_work_group
    __SYCL2020_DEPRECATED("use 'sycl::memory_scope_work_group' instead") =
        memory_scope::work_group;
__SYCL_INLINE_CONSTEXPR memory_scope memory_scope_device __SYCL2020_DEPRECATED(
    "use 'sycl::memory_scope_device' instead") = memory_scope::device;
__SYCL_INLINE_CONSTEXPR memory_scope memory_scope_system __SYCL2020_DEPRECATED(
    "use 'sycl::memory_scope_system' instead") = memory_scope::system;

#ifndef __SYCL_DEVICE_ONLY__
namespace detail {

__SYCL2020_DEPRECATED(
    "use 'sycl::detail::getStdMemoryOrder(sycl::memory_order)' instead")
static inline constexpr std::memory_order
getStdMemoryOrder(::cl::sycl::ext::oneapi::memory_order order) {
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

} // namespace detail
#endif // __SYCL_DEVICE_ONLY__

} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
