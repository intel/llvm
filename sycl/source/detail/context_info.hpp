//==---------------- context_info.hpp - SYCL context -----------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <detail/context_impl.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/info_desc_helpers.hpp>
#include <sycl/info/info_desc.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

template <typename Param>
typename Param::return_type get_context_info(RT::PiContext Ctx,
                                             const plugin &Plugin) {
  static_assert(is_context_info_desc<Param>::value,
                "Invalid context information descriptor");
  typename Param::return_type Result = 0;
  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin.call<PiApiKind::piContextGetInfo>(Ctx, PiInfoCode<Param>::value,
                                           sizeof(Result), &Result, nullptr);
  return Result;
}

// Specialization for atomic_memory_order_capabilities, PI returns a bitfield
template <>
std::vector<sycl::memory_order>
get_context_info<info::context::atomic_memory_order_capabilities>(
    RT::PiContext Ctx, const plugin &Plugin) {
  pi_memory_order_capabilities Result;
  Plugin.call<PiApiKind::piContextGetInfo>(
      Ctx, PiInfoCode<info::context::atomic_memory_order_capabilities>::value,
      sizeof(Result), &Result, nullptr);
  return readMemoryOrderBitfield(Result);
}
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
